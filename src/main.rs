use std::fs::File;

use get_kmers::get_sunk_positions;
use io::Fasta;
use polars::prelude::*;

mod io;
mod get_kmers;

fn main() -> eyre::Result<()> {
    let kmer_size = 20;
    let asm_fh = Fasta::new("test/input/all.fa")?;
    let ont_fh = Fasta::new("test/input/all_ONT.fa")?;
    let mut df_asm_sunks = get_sunk_positions(asm_fh, kmer_size, true)?;
    let mut df_ont_sunks = get_sunk_positions(ont_fh, kmer_size, true)?;
    let mut file_sunks_asm = File::create("asm_sunks.tsv").expect("could not create file");
    let mut file_sunks_ont = File::create("ont_sunks.tsv").expect("could not create file");

    CsvWriter::new(&mut file_sunks_asm)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df_asm_sunks)?;

    
    CsvWriter::new(&mut file_sunks_ont)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df_ont_sunks)?;
    Ok(())
}
