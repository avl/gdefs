#[macro_use]
extern crate anyhow;
extern crate lazy_static;
extern crate byteorder;

extern crate ring;

use ring::digest;

extern crate log;
extern crate indexmap;

use std::collections::{ VecDeque};
use anyhow::{Result};
use indexmap::set::IndexSet;
use indexmap::map::IndexMap;
use std::convert::TryInto;
use byteorder::{LittleEndian, ByteOrder};
use ring::digest::Context;
use env_logger::Env;
use std::fmt::{Debug, Formatter};

pub const SUB_BLOCK_SIZE:usize = 32;//4096
pub const BLOCK_SIZE:usize = 512;//262144;
pub const NUM_PEB :usize = 4;
pub const REDUNDANCY:u8=2;

pub const PEB_SEGMENT_HEADER_SIZE:usize=128;

pub const PEB_SEGMENT_HEADER_META_SIZE:usize=83;
pub const PEB_SEGMENT_HEADER_CHECKSUM_SIZE:usize=32;

pub const PHYSICAL_HEADER_SIZE:usize=32;


#[derive(Clone,Debug)]
struct PebSegmentHeader {
    filename: String,      //0  -> 64 (first 1 byte is size. Max size 63 byte)
    cycle: u64,          //64 -> 72
    start_offset: u32,     //72 -> 76
    datasize: u32,             //76 -> 80
    peb: u16,              //80 -> 82
    redundancy_channel: u8,//82 -> 83
}


#[derive(Debug)]
pub struct CorrectingBlockDevice<BD: Flash +Debug> {
    bd:BD
}
fn get_size_including_checksums(datasize:usize) -> usize {
    let num_sub_blocks = (datasize+SUB_BLOCK_SIZE-1)/SUB_BLOCK_SIZE;
    datasize + num_sub_blocks*PHYSICAL_HEADER_SIZE
}
impl<BD: Flash +Debug> CorrectingBlockDevice<BD> {

    fn correcting_blockdevice_get_erase_block_size(&self) -> usize {
        return BLOCK_SIZE;
    }

    fn correcting_blockdevice_write(&mut self, offset: usize, data: &[u8]) -> Result<usize> {
        println!("logical_write: {}..{}",offset,offset+data.len());
        if data.len()==0 {
            bail!("Can't write zero bytes")
        }
        let mut writeoffset = offset;
            for suboffset in (0..data.len()).step_by(SUB_BLOCK_SIZE) {

            writeoffset += self.write_impl(offset+suboffset,&data[suboffset..(suboffset+SUB_BLOCK_SIZE).min(data.len())])?;
        }
        Ok(writeoffset-offset)
    }
    fn write_impl(&mut self, offset: usize, data: &[u8]) -> Result<usize> {
        println!("bare_raw_write: {}..{}",offset,offset+data.len());
        if data.len()==0 {
            bail!("Can't write zero bytes")
        }
        let cksum = calc_simple_checksum(data);
        self.bd.flash_write(offset, data)?;
        self.bd.flash_write(offset+data.len(), &cksum)?;
        Ok(data.len()+PHYSICAL_HEADER_SIZE)

    }
    fn correcting_blockdevice_read(&self, offset: usize, data: &mut [u8]) -> Result<usize> {
        if data.len()==0 {
            bail!("Can't read zero bytes")
        }
        let mut read_offset = offset;
        for suboffset in (0..data.len()).step_by(SUB_BLOCK_SIZE) {

            let datalen = data.len();
            read_offset += self.read_impl(read_offset, &mut data[suboffset..(suboffset+SUB_BLOCK_SIZE).min(datalen)])?;
        }
        Ok(read_offset-offset)
    }
    fn read_impl(&self, offset: usize, data: &mut [u8]) -> Result<usize> {
        println!("raw_read: {}..{}",offset,offset+data.len());
        if data.len()==0 {
            bail!("Can't read zero bytes")
        }
        let datalen = data.len();
        self.bd.flash_read(offset, &mut data[..])?;
        let mut checksum = [0u8;PHYSICAL_HEADER_SIZE];
        self.bd.flash_read(offset+datalen, &mut checksum)?;
        if calc_simple_checksum(data) == checksum {
            return Ok(datalen+PHYSICAL_HEADER_SIZE);
        }
        bail!("Checksum error")
    }

    /// Erases the entire block
    fn correcting_blockdevice_erase(&mut self, block:u16) -> Result<()> {
        let offset = (block as usize)*BLOCK_SIZE;
        let size = BLOCK_SIZE;
        println!("raw_erase: {}..{}",offset,offset+size);
        self.bd.flash_erase(offset, size)?;
        Ok(())
    }

    fn correcting_blockdevice_read_smart(&self, candidates: &[usize], data:&mut [u8]) -> Result<()> {
        let datalen = data.len();
        let mut readoffset = 0;
        for suboffset in (0..datalen).step_by(SUB_BLOCK_SIZE) {
            let mut success=false;

            let curdata = &mut data[suboffset..(suboffset+SUB_BLOCK_SIZE).min(datalen)];
            for cand_offset in candidates {
                match self.read_impl(cand_offset+readoffset, curdata) {
                    Ok(t) => {
                        success = true;
                        break;
                    }
                    Err(_) => {}
                }
            }
            readoffset += curdata.len() + PHYSICAL_HEADER_SIZE;
            if !success {
                bail!("Could not reassemble sub block at {}",suboffset);
            }
        }
        Ok(())
    }
}

impl<BD: Flash +Debug> Debug for FileSystem<BD> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSystem")
            .field("cycle", &self.cycle)
            .field("files", &self.files)
            .field("free_peb", &self.free_peb)
            .field("bad_peb", &self.bad_peb)
            .finish()
    }
}
impl Flash for Vec<u8> {
    fn flash_get_erase_block_size(&self) -> usize {
        return BLOCK_SIZE;
    }

    fn flash_write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        self[offset..offset+data.len()].copy_from_slice(data);
        Ok(())
    }
    fn flash_read(&self, offset: usize, data: &mut [u8]) -> Result<()> {
        data.copy_from_slice(&self[offset..offset+data.len()]);
        Ok(())
    }

    fn flash_erase(&mut self, offset: usize, size: usize) -> Result<()> {
        for x in offset..offset+size {
            self[x] = 0;
        }
        Ok(())
    }
}

impl PebSegmentHeader {
    pub fn free_payload_space(&self) -> usize {
        BLOCK_SIZE.saturating_sub(self.start_offset as usize + get_size_including_checksums(self.datasize as usize) as usize+PEB_SEGMENT_HEADER_SIZE)
    }
    pub fn from_buf(buf:&[u8]) -> Result<PebSegmentHeader> {
        if buf.len() != PEB_SEGMENT_HEADER_META_SIZE {
            bail!("Can't create PebSegmentHeader, wrong size of buffer")
        }
        let namelen = buf[0] as usize;
        if namelen > 63 {
            bail!("Corrupt name length");
        }
        let filename = std::str::from_utf8(&buf[1..1+namelen])?;
        Ok(PebSegmentHeader{
            filename: filename.to_string(),
            cycle: LittleEndian::read_u64(&buf[64..72]),
            start_offset: LittleEndian::read_u32(&buf[72..76]),
            datasize: LittleEndian::read_u32(&buf[76..80]),
            peb: LittleEndian::read_u16(&buf[80..82]),
            redundancy_channel: buf[82],
        })

    }
    pub fn to_buf(&self) -> [u8;PEB_SEGMENT_HEADER_META_SIZE] {
        let mut buf = [0u8;PEB_SEGMENT_HEADER_META_SIZE];
        let file_bytes = self.filename.as_bytes();
        assert!(file_bytes.len() <= 63);
        buf[0] = file_bytes.len() as u8;
        buf[1..1+file_bytes.len()].copy_from_slice(file_bytes);

        LittleEndian::write_u64(&mut buf[64..72], self.cycle);
        LittleEndian::write_u32(&mut buf[72..76], self.start_offset);
        LittleEndian::write_u32(&mut buf[76..80], self.datasize);
        LittleEndian::write_u16(&mut buf[80..82], self.peb);
        buf[82] = self.redundancy_channel;
        buf
    }
}

#[derive(Debug)]
struct FileData {
    redundancies: Vec<PebSegmentHeader>
}

pub trait Flash {
    fn flash_get_erase_block_size(&self) -> usize;
    /// If write is smaller than physical block size, 0 padding should be used by driver
    fn flash_write(&mut self, offset: usize, data: &[u8]) -> Result<()>;
    fn flash_read(&self, offset: usize, data:&mut [u8]) -> Result<()>;
    fn flash_erase(&mut self, offset: usize, size: usize) -> Result<()>;
}

fn calc_simple_checksum(header:&[u8]) -> [u8;32] {

    let mut ctx = Context::new(&digest::SHA256);
    ctx.update(header);

    let hash = ctx.finish();
    let result:[u8;32] = hash.as_ref().try_into().unwrap();
    result
}

fn calc_checksum(header:&[u8], data:&[u8]) -> [u8;32] {

    let mut ctx = Context::new(&digest::SHA256);
    ctx.update(header);
    ctx.update(data);

    let hash = ctx.finish();
    let result:[u8;32] = hash.as_ref().try_into().unwrap();
    result
}
pub struct FileSystem<BD: Flash +Debug> {
    cycle: u64,
    files: IndexMap<String, FileData>,
    free_peb: VecDeque<u16>,
    bad_peb: IndexSet<u16>,
    block_device: CorrectingBlockDevice<BD>,
}


fn start_of_peb(peb:u16) -> usize {
    (peb as usize) * BLOCK_SIZE
}

impl<BD: Flash +Debug> FileSystem<BD> {

    fn scan_advance_candidate(peb: u16, offset:&mut usize, old_candidate:&Option<PebSegmentHeader>, bd: &BD) -> Result<PebSegmentHeader> {
        let mut headerbuf = vec![0u8;PEB_SEGMENT_HEADER_SIZE];
        bd.flash_read(start_of_peb(peb) + *offset, &mut headerbuf)?;
        let header = PebSegmentHeader::from_buf(&headerbuf[32..32+PEB_SEGMENT_HEADER_META_SIZE]);
        if let Ok(new_candidate) = header {
            if let Some(old_candidate) = old_candidate {
                if new_candidate.cycle < old_candidate.cycle {
                    bail!("Bad cycle");
                }
                if old_candidate.filename != new_candidate.filename {
                    bail!("Bad filename");
                }
            }
            if calc_simple_checksum(&headerbuf[32..32+PEB_SEGMENT_HEADER_META_SIZE]) == headerbuf[0..32] {
                return Ok(new_candidate);
            } else {
                bail!("Checksum error")
            }
        } else {
            bail!("Invalid header")
        }
    }

    fn scan_peb(files: &mut IndexMap<String, FileData>, block_device: &BD, peb:u16, max_cycle: &mut u64) -> Result<()> {
        let mut offset = 0usize;
        let mut candidate = None;
        loop {
            match Self::scan_advance_candidate(peb, &mut offset, &candidate, &block_device) {
                Ok(new_candidate) => {
                    *max_cycle = new_candidate.cycle.max(*max_cycle);
                    offset += get_size_including_checksums(new_candidate.datasize as usize) as usize+PEB_SEGMENT_HEADER_SIZE;
                    candidate = Some(new_candidate);
                }
                Err(err) => {
                    log::debug!("Error during scan: {:?}",err);
                    break;
                }
            }

        }
        if let Some(candidate) = candidate {
            let filename = candidate.filename.clone();
            files.entry(filename).or_insert(FileData{redundancies:vec![]}).redundancies.push(candidate);
            return Ok(());
        } else {
            bail!("No usable entry found")
        }
    }

    pub fn into_inner(self) -> BD {
        self.block_device.bd
    }
    pub fn inner(&self) -> &BD {
        &self.block_device.bd
    }
    pub fn inner_mut(&mut self) -> &mut BD {
        &mut self.block_device.bd
    }
    pub fn new(bd: BD) -> Result<FileSystem<BD>> {
        let mut max_cycle = 0;
        let mut files = IndexMap::new();
        let mut free_peb = VecDeque::new();
        for peb in 0..NUM_PEB as u16 {
            match Self::scan_peb(&mut files, &bd, peb, &mut max_cycle) {
                Ok(()) => {log::debug!("PEB {}: OK!", peb)}
                Err(err) => {
                    free_peb.push_back(peb);
                    log::debug!("PEB {}: {:?}",peb,err)
                }
            }
        }
        for (_filename,meta) in &mut files {
            let max_cycle = meta.redundancies.iter().map(|x|x.cycle).max();
            meta.redundancies = meta.redundancies.iter().cloned().filter(|x|Some(x.cycle)==max_cycle).collect();
        }
        free_peb.rotate_left(max_cycle as usize % free_peb.len()); //Just to 'randomize' where we start writing
        Ok(FileSystem {
            cycle: max_cycle,
            files,
            free_peb,
            bad_peb: IndexSet::new(),
            block_device: CorrectingBlockDevice{bd},
        })
    }

    pub fn read_file(&self, filename: &str) -> Result<Vec<u8>> {
        let mut candidates = Vec::new();
        let mut filesize = None;
        if let Some(data) = self.files.get(filename) {
            for pebdata in &data.redundancies {
                if let Some(filesize) = filesize {
                    if filesize != pebdata.datasize {
                        bail!("Unexpected error: Multiple different, correctly checksummed, sizes for file {:?}",filename);
                    }
                } else {
                    filesize = Some(pebdata.datasize);
                }
                candidates.push(start_of_peb(pebdata.peb)+pebdata.start_offset as usize+PEB_SEGMENT_HEADER_SIZE+PHYSICAL_HEADER_SIZE);
            }
        } else {
            bail!("File not found")
        }
        if let Some(filesize) = filesize {
            let mut retval = vec![0u8;filesize as usize];
            println!("CAndidates: {:?}",candidates);
            self.block_device.correcting_blockdevice_read_smart(&candidates, &mut retval)?;
            Ok(retval)
        } else {
            bail!("The set of redundant copies for the file was empty!")
        }
    }

    fn modify_file(file: &mut PebSegmentHeader, cycle: u64, data: &[u8], block_device:&mut CorrectingBlockDevice<BD>) -> Result<()> {
        if data.len() > std::u32::MAX as usize {
            bail!("Data chunk too big to write to file")
        }
        let mut header = [0u8;PEB_SEGMENT_HEADER_SIZE];

        let write_start = file.start_offset as usize +get_size_including_checksums(file.datasize as usize) as usize + PEB_SEGMENT_HEADER_SIZE+PHYSICAL_HEADER_SIZE;
        let mut file_copy = file.clone();

        file_copy.start_offset = write_start as u32;
        file_copy.cycle = cycle;
        file_copy.datasize = data.len() as u32;
        if file_copy.start_offset as usize > BLOCK_SIZE {
            bail!("Out of space to modify file")
        }
        let meta_buf = file_copy.to_buf();
        let checksum = calc_simple_checksum(&meta_buf);
        header[0..32].copy_from_slice(&checksum);
        header[32..32+PEB_SEGMENT_HEADER_META_SIZE].copy_from_slice(&meta_buf);

        let write_start_fs_offset = start_of_peb(file_copy.peb)+write_start as usize;
        block_device.correcting_blockdevice_write(write_start_fs_offset, &header)?;
        block_device.correcting_blockdevice_write(write_start_fs_offset+PEB_SEGMENT_HEADER_SIZE+PHYSICAL_HEADER_SIZE, data)?;
        *file = file_copy;
        Ok(())
    }
    fn initialize_peb(&mut self, peb:u16, filename: &str, cycle:u64, data: &[u8], redundancy: u8) -> Result<PebSegmentHeader> {
        if filename.as_bytes().len() > 63 {
            bail!("Filename contains more than 63 bytes when formatted as utf8");
        }
        if data.len() + PEB_SEGMENT_HEADER_SIZE > BLOCK_SIZE {
            bail!("File is too large to write to this filesystem");
        }
        self.block_device.correcting_blockdevice_erase(peb)?;

        let header = PebSegmentHeader {
            filename: filename.to_string(),
            cycle,
            start_offset: 0,
            datasize: data.len() as u32,
            peb,
            redundancy_channel: redundancy
        };
        let meta_header_buf = header.to_buf();
        let cksum = calc_simple_checksum(&meta_header_buf);
        let mut header_buf = [0u8;PEB_SEGMENT_HEADER_SIZE];
        header_buf[0..32].copy_from_slice(&cksum);
        header_buf[32..32+PEB_SEGMENT_HEADER_META_SIZE].copy_from_slice(&meta_header_buf);

        let write_start_fs_offset = start_of_peb(peb);
        self.block_device.correcting_blockdevice_write(write_start_fs_offset, &header_buf)?;
        self.block_device.correcting_blockdevice_write(write_start_fs_offset+PEB_SEGMENT_HEADER_SIZE+PHYSICAL_HEADER_SIZE, data)?;

        Ok(header)
    }
    pub fn erase_file(&mut self, filename:&str) -> Result<()> {
        if let Some(file) = self.files.get(filename) {
            for redund in &file.redundancies {
                match self.block_device.correcting_blockdevice_erase(redund.peb) {
                    Ok(()) => {},
                    Err(err) => log::warn!("Could not erase block {}: {:?}", redund.peb, err)
                }
            }
            self.files.remove(filename);
        }
        Ok(())
    }
    pub fn get_num_free_blocks(&self) -> usize {
        self.free_peb.len()
    }

    pub fn ls(&self) -> impl Iterator<Item=&str> {
        self.files.keys().map(|x|x.as_str())
    }

    pub fn store_file(&mut self, filename:&str, data: &[u8]) -> Result<()> {
        if self.free_peb.len() < REDUNDANCY.into() {
            bail!("Filesystem is full")
        }
        println!("Store file");
        self.cycle += 1;

        let mut pebs_which_were_full_and_should_be_reused_if_all_goes_well = Vec::new();

        let mut failed=false;
        let mut pebs_found = vec![];
        for redundancy_channel in 0..REDUNDANCY {
            println!("Files: {:?}",&self.files);
            if let Some(existing_file) = self.files.get_mut(filename)
                .map(|x: &mut FileData| x.redundancies.iter_mut()
                    .find(|x| x.redundancy_channel == redundancy_channel)).flatten() {
                if existing_file.free_payload_space() < data.len() {
                    pebs_which_were_full_and_should_be_reused_if_all_goes_well.push(existing_file.peb);
                    failed = true;
                    continue;
                }
                println!("Modifying channel {}",redundancy_channel);
                match Self::modify_file(existing_file, self.cycle, data, &mut self.block_device) {
                    Ok(()) => {
                        pebs_found.push(existing_file.peb);
                    }
                    Err(err) => {
                        log::warn!("Failed to modify_file peb {}: {:?}. Not giving up, trying to rewrite entire file.",existing_file.peb,err);
                        pebs_which_were_full_and_should_be_reused_if_all_goes_well.push(existing_file.peb);
                        failed = true;
                    }
                }
            }
        }
        if pebs_found.len() != REDUNDANCY as usize {
            pebs_which_were_full_and_should_be_reused_if_all_goes_well.extend(pebs_found);
            failed = true;
        }
        if !failed {
            return Ok(());
        }
        self.cycle += 1;

        let mut new_pebs = Vec::new();

        for redundancy in 0..REDUNDANCY {
            loop {
                let peb = self.free_peb.pop_front().ok_or_else(||anyhow!("Out of disk space"))?;
                match self.initialize_peb(peb,filename,self.cycle, data,redundancy) {
                    Ok(peb) => {
                        new_pebs.push(peb);
                        break;
                    }
                    Err(err) => {
                        log::warn!("Failed to write peb {}: {:?}",peb,err);
                        continue;
                    }
                }
            }
        }
        self.files.insert(filename.to_string(), FileData {
            redundancies: new_pebs
        });
        self.free_peb.extend(pebs_which_were_full_and_should_be_reused_if_all_goes_well);
        Ok(())
    }
}



#[cfg(test)]
mod tests {
    extern crate env_logger;

    use crate::{BLOCK_SIZE, NUM_PEB, FileSystem, start_of_peb};
    use anyhow::Result;
    use self::env_logger::Env;

    fn make_flash() -> Vec<u8> {
        vec![0u8; NUM_PEB * BLOCK_SIZE]
    }
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn setup() {
        INIT.call_once(||env_logger::from_env(Env::default().default_filter_or("trace")).init());
    }

    #[test]
    fn basic_fs_tests() -> Result<()> {
        setup();
        let mut flash = make_flash();

        let mut fs = FileSystem::new(flash)?;

        fs.store_file("test.txt", &[42])?;

        let files: Vec<_> = fs.ls().collect();
        assert_eq!(files, vec!["test.txt"]);

        let flash = fs.into_inner();
        let mut fs = FileSystem::new(flash)?;
        let files: Vec<_> = fs.ls().collect();
        assert_eq!(files, vec!["test.txt"]);
        let readback = fs.read_file("test.txt")?;
        assert_eq!(readback, vec![42]);

        Ok(())
    }

    fn make_filesystem_with_test_txt() -> Result<FileSystem<Vec<u8>>> {
        let mut flash = make_flash();
        let mut fs = FileSystem::new(flash)?;

        fs.store_file("test.txt", &[1, 2, 3, 4])?;
        Ok(fs)
    }

    fn fill(slice:&mut [u8],data:u8) {
        for item in slice {
            *item = data;
        }
    }
    #[test]
    fn basic_read_test() -> Result<()> {
        setup();
        let mut fs = make_filesystem_with_test_txt()?;
        let readback = fs.read_file("test.txt")?;
        println!("Read: {:?}",readback);
        assert_eq!(readback, vec![1,2,3,4]);
        Ok(())
    }
        #[test]
    fn basic_redundancy_test() -> Result<()> {
        setup();
        let mut fs = make_filesystem_with_test_txt()?;
        fill(&mut fs.inner_mut()[start_of_peb(0)..start_of_peb(0)+BLOCK_SIZE], 0);
        println!("Fs: {:#?}", fs);
        let readback = fs.read_file("test.txt")?;
        println!("Read: {:?}",readback);
        assert_eq!(readback, vec![1,2,3,4]);
        Ok(())
    }
    #[test]
    fn modify_file_test() -> Result<()> {
        setup();
        let mut fs = make_filesystem_with_test_txt()?;
        let free_blocks = fs.get_num_free_blocks();
        println!("Modify");
        fs.store_file("test.txt", &[5,6,7,8])?;
        let readback = fs.read_file("test.txt")?;
        println!("fs: {:#?}",fs);
        assert_eq!(readback, vec![5,6,7,8]);
        assert_eq!(free_blocks, fs.get_num_free_blocks());
        Ok(())
    }
    #[test]
    fn modify_repeatedly_test() {
        todo!("Implement a test to check that a new PEB is allocated when the first PEB is written")
    }
    #[test]
    fn future_tests() {
        todo!("Fuzz test. Also, make BLOCK DEVICE params configurable")
    }
}