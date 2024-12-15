use core::str;
use std::collections::{HashMap, HashSet};

use io::Fasta;
use itertools::Itertools;
use kmers::{self, Kmer};
use rayon::prelude::*;

mod io;

pub type KmerPositions = HashMap<String, HashMap<Kmer, Vec<usize>>>;

/// Extract all k-mers positions from a given sequence. 
/// * Mimic behavior of jellyfish in counting canonical kmers
///     * `jellyfish -C -m kmer_size`
///     * https://www.genome.umd.edu/docs/JellyfishUserGuide.pdf
///         * 1.1.1 Counting k-mers in sequencing reads
fn get_kmer_indices(
    fasta: &str,
    name: &str,
    len: u64,
    kmer_size: usize,
) -> eyre::Result<HashMap<Kmer, Vec<usize>>> {
    let mut fh = Fasta::new(fasta)?;
    let rec = fh.fetch(name, 1, len.try_into()?)?;
    let mut indices: HashMap<Kmer, Vec<usize>> = HashMap::new();
    Kmer::with_many_both_pos(kmer_size, rec.sequence(), |pos, x, y| {
        indices
            .entry(x.clone())
            .or_insert_with(Vec::new)
            .push(pos + 1 - kmer_size);
        indices
            .entry(y.clone())
            .or_insert_with(Vec::new)
            .push(pos + 1 - kmer_size);
    });
    Ok(indices)
}

fn get_kmer_positions(
    fasta: Fasta,
    kmer_size: usize,
    kmer_cnt: usize,
) -> eyre::Result<KmerPositions> {
    let all_seq_lens: Vec<(String, u64)> = fasta
        .index
        .as_ref()
        .iter()
        .map(|rec| {
            (
                String::from_utf8(rec.name().to_vec()).unwrap(),
                rec.length(),
            )
        })
        .collect();
    let mut all_kmer_indices: HashMap<String, HashMap<Kmer, Vec<usize>>> = all_seq_lens
        .into_par_iter()
        .map(|(name, len)| {
            let kmer_indices =
                get_kmer_indices(fasta.fname.to_str().unwrap(), &name, len, kmer_size).unwrap();
            (name, kmer_indices)
        })
        .collect();
    
    // Sum up kmer counts across all sequences.
    let mut kmer_cnts: HashMap<Kmer, usize> =
        all_kmer_indices.values().fold(HashMap::new(), |mut a, b| {
            for (kmer, pos) in b.iter() {
                *a.entry(kmer.clone()).or_default() += pos.len()
            }
            a
        });
    kmer_cnts.retain(|_, cnt| *cnt == kmer_cnt);

    for (_, kmers) in all_kmer_indices.iter_mut() {
        // Filter kmers to ones that have desired count.
        kmers.retain(|k, _| kmer_cnts.contains_key(k));
        
        // TODO: There's probably a better way to do this.
        let mut keep_list: HashSet<Kmer> = HashSet::new();
        let mut remove_list: HashSet<Kmer> = HashSet::new();
        for (kmer, _) in kmers.iter() {
            // Does this kmer have a revcomp version in the kmers map?
            // If so, remove it to avoid double counting.
            let rev_comp_kmer = kmer.rev_comp(kmer_size);
            if !remove_list.contains(&rev_comp_kmer) && kmers.contains_key(&rev_comp_kmer) && !keep_list.contains(&rev_comp_kmer) {
                keep_list.insert(kmer.clone());
                remove_list.insert(rev_comp_kmer);
            }
        }
        // Remove kmers that have a revcomp.
        kmers.retain(|k, _| !remove_list.contains(k));
    }
    Ok(all_kmer_indices)
}

fn main() -> eyre::Result<()> {
    let kmer_size = 20;
    let fh = Fasta::new("test/input/all.fa")?;
    let sunks = get_kmer_positions(fh, kmer_size, 1)?;
    let mut n_kmers = 0;
    for (seq_name, kmer_positions) in sunks.iter() {
        for (kmer, positions) in kmer_positions
            .iter()
            .sorted_by(|a, b| a.1.first().cmp(&b.1.first()))
        {
            // TODO: Keep track of id here.
            let kmer_seq = kmer.render(kmer_size);
            for pos in positions {
                // println!("{seq_name}\t{pos}\t{kmer_seq}\t",)
            }
            n_kmers += 1;
        }
    }
    println!("{n_kmers}");
    Ok(())
}
