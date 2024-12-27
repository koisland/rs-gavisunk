use std::fs::File;

use assign_read_ctg::assign_read_to_ctg_w_ort;
use get_kmers::get_sunk_positions;
use io::Fasta;
use map_kmers::map_sunks_to_reads;
use polars::prelude::*;

mod assign_read_ctg;
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
    let mut file_sunks_asm = File::create("asm_sunks.tsv")?;

    log::info!("Mapping assembly SUNKs to reads.");
    let mut df_read_sunks = map_sunks_to_reads(ont_fh, &df_asm_sunks)?;
    let mut file_sunks_reads = File::create("read_sunks.tsv")?;

    log::info!("Assigning reads to assembly contigs.");
    let mut df_best_reads_asm = assign_read_to_ctg_w_ort(&df_read_sunks, None, None)?;
    let mut file_best_reads_asm = File::create("read_ctg_mapping.tsv")?;

    CsvWriter::new(&mut file_sunks_asm)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df_asm_sunks)?;
    CsvWriter::new(&mut file_sunks_reads)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df_read_sunks)?;
    CsvWriter::new(&mut file_best_reads_asm)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df_best_reads_asm)?;

    Ok(())
}
