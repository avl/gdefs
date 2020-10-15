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

pub const BLOCK_SIZE:usize = 262144;
pub const NUM_PEB :usize = 1;
pub const REDUNDANCY:u8=1;

pub const PEB_SEGMENT_HEADER_SIZE:usize=128;

pub const PEB_SEGMENT_HEADER_META_SIZE:usize=83;
pub const PEB_SEGMENT_HEADER_CHECKSUM_SIZE:usize=32;





#[derive(Clone,Debug)]
struct PebSegmentHeader {
    filename: String,      //0  -> 64 (first 1 byte is size. Max size 63 byte)
    cycle: u64,          //64 -> 72
    start_offset: u32,     //72 -> 76
    datasize: u32,             //76 -> 80
    peb: u16,              //80 -> 82
    redundancy_channel: u8,//82 -> 83
}

impl BlockDevice for &mut Vec<u8> {
    fn get_erase_block_size(&self) -> usize {
        return BLOCK_SIZE;
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        self[offset..offset+data.len()].copy_from_slice(data);
        Ok(())
    }

    fn read(&self, offset: usize, data: &mut [u8]) -> Result<()> {
        data.copy_from_slice(&self[offset..offset+data.len()]);
        Ok(())
    }

    fn erase(&mut self, offset: usize, size: usize) -> Result<()> {
        for x in offset..offset+size {
            self[x] = 0;
        }
        Ok(())
    }
}

impl PebSegmentHeader {
    pub fn read(&self, bd:&dyn BlockDevice) -> Result<Vec<u8>> {
        let mut headerbuf = vec![0u8; PEB_SEGMENT_HEADER_SIZE];
        bd.read(self.start_offset as usize,&mut headerbuf)?;
        let header = PebSegmentHeader::from_buf(&headerbuf[32..32+PEB_SEGMENT_HEADER_META_SIZE])?;
        if header.cycle != self.cycle {
            bail!("In sanity check, wrong cycle");
        }
        if header.datasize != self.datasize {
            bail!("In sanity check, wrong size");
        }

        let mut payloaddatabuf = vec![0u8; self.datasize as usize];

        bd.read(self.start_offset as usize +PEB_SEGMENT_HEADER_SIZE,&mut payloaddatabuf[..])?;
        let calculated_checksum = calc_checksum(&headerbuf[32..32+PEB_SEGMENT_HEADER_META_SIZE], &payloaddatabuf);

        if calculated_checksum==headerbuf[0..32] {
            Ok(payloaddatabuf)
        } else {
            bail!("Checksum error reading file from peb {}",self.peb);
        }
    }
    pub fn free_payload_space(&self) -> usize {
        BLOCK_SIZE.saturating_sub(self.start_offset as usize + self.datasize as usize+PEB_SEGMENT_HEADER_SIZE)
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
struct FileData {
    redundancies: Vec<PebSegmentHeader>
}

pub trait BlockDevice {
    fn get_erase_block_size(&self) -> usize;
    /// If write is smaller than physical block size, 0 padding should be used by driver
    fn write(&mut self, offset: usize, data: &[u8]) -> Result<()>;
    fn read(&self, offset: usize, data:&mut [u8]) -> Result<()>;
    fn erase(&mut self, offset: usize, size: usize) -> Result<()>;
}

fn calc_checksum(header:&[u8], data:&[u8]) -> [u8;32] {

    let mut ctx = Context::new(&digest::SHA256);
    ctx.update(header);
    ctx.update(data);

    let hash = ctx.finish();
    let result:[u8;32] = hash.as_ref().try_into().unwrap();
    result
}
pub struct FileSystem<BD:BlockDevice> {
    cycle: u64,
    files: IndexMap<String, FileData>,
    free_peb: VecDeque<u16>,
    bad_peb: IndexSet<u16>,
    block_device: BD,
}


fn start_of_peb(peb:u16) -> usize {
    (peb as usize) * BLOCK_SIZE
}

impl<BD:BlockDevice> FileSystem<BD> {

    fn scan_advance_candidate(peb: u16, offset:&mut usize, old_candidate:&Option<PebSegmentHeader>, bd: &BD) -> Result<PebSegmentHeader> {
        let mut headerbuf = vec![0u8;PEB_SEGMENT_HEADER_SIZE];
        bd.read(start_of_peb(peb) + *offset,&mut headerbuf)?;
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
            let mut databuf=vec![0u8; new_candidate.datasize as usize];
            bd.read(start_of_peb(peb) + *offset+PEB_SEGMENT_HEADER_SIZE, &mut databuf)?;
            if calc_checksum(&headerbuf[32..32+PEB_SEGMENT_HEADER_META_SIZE],&databuf) == headerbuf[0..32] {
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
                    offset += new_candidate.datasize as usize+PEB_SEGMENT_HEADER_SIZE;
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
        free_peb.rotate_left((max_cycle % (NUM_PEB as u64)) as usize); //Just to 'randomize' where we start writing
        Ok(FileSystem {
            cycle: max_cycle,
            files,
            free_peb,
            bad_peb: IndexSet::new(),
            block_device: bd
        })
    }

    pub fn read_file(&self, filename: &str) -> Result<Vec<u8>> {
        if let Some(data) = self.files.get(filename) {
            for pebdata in &data.redundancies {
                if let Ok(data) = pebdata.read(&self.block_device) {
                    return Ok(data);
                }
            }
            bail!("None of the {} copies of the file was uncorrupted",REDUNDANCY)
        } else {
            bail!("File not found")
        }
    }

    fn modify_file(file: &mut PebSegmentHeader, cycle: u64, data: &[u8], block_device:&mut BD) -> Result<()> {
        if data.len() > std::u32::MAX as usize {
            bail!("Data chunk too big to write to file")
        }
        let mut header = [0u8;PEB_SEGMENT_HEADER_SIZE];

        let write_start = file.start_offset as usize +file.datasize as usize + PEB_SEGMENT_HEADER_SIZE;
        let mut file_copy = file.clone();

        file_copy.start_offset = (file.start_offset as usize +file.datasize as usize+PEB_SEGMENT_HEADER_SIZE) as u32;
        file_copy.cycle = cycle;
        file_copy.datasize = data.len() as u32;
        if file_copy.start_offset as usize > BLOCK_SIZE {
            bail!("Out of space to modify file")
        }
        let meta_buf = file_copy.to_buf();
        let checksum = calc_checksum(&meta_buf, data);
        header[0..32].copy_from_slice(&checksum);
        header[32..32+PEB_SEGMENT_HEADER_META_SIZE].copy_from_slice(&meta_buf);

        let write_start_fs_offset = start_of_peb(file_copy.peb)+write_start as usize;
        block_device.write(write_start_fs_offset,&header)?;
        block_device.write(write_start_fs_offset+PEB_SEGMENT_HEADER_SIZE,data)?;
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
        self.block_device.erase(start_of_peb(peb),BLOCK_SIZE)?;

        let header = PebSegmentHeader {
            filename: filename.to_string(),
            cycle,
            start_offset: 0,
            datasize: data.len() as u32,
            peb,
            redundancy_channel: redundancy
        };
        let meta_header_buf = header.to_buf();
        let cksum = calc_checksum(&meta_header_buf, data);
        let mut header_buf = [0u8;PEB_SEGMENT_HEADER_SIZE];
        header_buf[0..32].copy_from_slice(&cksum);
        header_buf[32..32+PEB_SEGMENT_HEADER_META_SIZE].copy_from_slice(&meta_header_buf);

        let write_start_fs_offset = start_of_peb(peb);
        self.block_device.write(write_start_fs_offset, &header_buf)?;
        self.block_device.write(write_start_fs_offset+PEB_SEGMENT_HEADER_SIZE,data)?;

        Ok(header)
    }
    pub fn erase_file(&mut self, filename:&str) -> Result<()> {
        if let Some(file) = self.files.get(filename) {
            for redund in &file.redundancies {
                match self.block_device.erase(start_of_peb(redund.peb),BLOCK_SIZE) {
                    Ok(()) => {},
                    Err(err) => log::warn!("Could not erase block {}: {:?}", redund.peb, err)
                }
            }
            self.files.remove(filename);
        }
        Ok(())
    }

    pub fn ls(&self) -> impl Iterator<Item=&str> {
        self.files.keys().map(|x|x.as_str())
    }

    pub fn store_file(&mut self, filename:&str, data: &[u8]) -> Result<()> {
        if self.free_peb.len() < REDUNDANCY.into() {
            bail!("Filesystem is full")
        }
        self.cycle += 1;

        let mut pebs_which_were_full_and_should_be_reused_if_all_goes_well = Vec::new();

        let mut failed=false;
        let mut pebs_found = vec![];
        for redundancy_channel in 0..3 {
            if let Some(existing_file) = self.files.get_mut(filename)
                .map(|x: &mut FileData| x.redundancies.iter_mut()
                    .find(|x| x.redundancy_channel == redundancy_channel)).flatten() {
                if existing_file.free_payload_space() < data.len() {
                    pebs_which_were_full_and_should_be_reused_if_all_goes_well.push(existing_file.peb);
                    failed = true;
                    continue;
                }
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

    use crate::{BLOCK_SIZE, NUM_PEB, FileSystem};
    use anyhow::Result;
    use self::env_logger::Env;

    fn make_flash() -> Vec<u8> {
        vec![0u8;NUM_PEB*BLOCK_SIZE]
    }
    #[test]
    fn create_fs() -> Result<()> {

        env_logger::from_env(Env::default().default_filter_or("trace")).init();
        let mut flash = make_flash();

        let mut fs = FileSystem::new(&mut flash)?;

        fs.store_file("test.txt", &[42])?;

        let files : Vec<_> = fs.ls().collect();
        println!("Ls: {:?}", files);

        println!("----------------------------------");
        println!("----------------------------------");
        println!("----------------------------------");
        let mut fs = FileSystem::new(&mut flash)?;
        let files : Vec<_> = fs.ls().collect();
        println!("Ls: {:?}", files);

        Ok(())
    }
}
