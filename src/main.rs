use std::fs::File;

use get_kmers::get_sunk_positions;
use io::Fasta;
use map_kmers::map_sunks_to_seqs;
use polars::prelude::*;

mod get_kmers;
mod io;
mod map_kmers;

fn main() -> eyre::Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()?;

    let kmer_size = 20;
    let asm_fh = Fasta::new("test/input/all.fa")?;
    let ont_fh = Fasta::new("test/input/all_ONT.fa")?;

    log::info!("Getting SUNK positions in assembly.");
    let mut df_asm_sunks = get_sunk_positions(asm_fh, kmer_size, true)?;
    let mut file_sunks_asm = File::create("asm_sunks.tsv").expect("could not create file");

    log::info!("Mapping SUNKs to reads.");
    let mut df_read_sunks = map_sunks_to_seqs(ont_fh, &df_asm_sunks)?;
    let mut file_sunks_reads = File::create("read_sunks.tsv").expect("could not create file");

    CsvWriter::new(&mut file_sunks_asm)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df_asm_sunks)?;
    CsvWriter::new(&mut file_sunks_reads)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df_read_sunks)?;

    Ok(())
}
