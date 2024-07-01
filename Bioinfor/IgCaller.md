---
toc: true
url: IgCaller
covercopy: © Karobben
priority: 10000
date: 2024-06-24 16:45:27
title: "IgCaller"
ytitle: "IgCaller"
description: "Reconstructing immunoglobulin gene rearrangements and oncogenic translocations from WGS, WES, and capture NGS data"
excerpt: "IgCaller is a python program designed to fully characterize the immunoglobulin gene rearrangements and oncogenic translocations in lymphoid neoplasms. It was originally developed to work with WGS data but it has been extended to work with WES and high-coverage, capture-based NGS data."
tags: [Antibody, Immunology, Genome, Bioinformatics, NGS]
category: [Biology, Bioinformatics, Software ]
cover: "https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41467-020-17095-7/MediaObjects/41467_2020_17095_Fig1_HTML.png"
thumbnail: "https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41467-020-17095-7/MediaObjects/41467_2020_17095_Fig1_HTML.png"
---


IgCaller is used extensively in immunology research to study B-cell receptor diversity and antibody generation mechanisms. Clinically, it helps identify clonal B-cell expansions, monitor minimal residual disease in leukemias and lymphomas, and analyze antibody responses to vaccines. Additionally, it supports therapeutic antibody development by identifying candidate antibodies from strong immune responses.

It is an [open-source](https://github.com/ferrannadeu/IgCaller) tool designed to study human B cell Ig gene rearrangements. According to the documentation, it only supports the human hg19 or hg38 genome as the input reference, so the application of this tool is limited to humans. It requires selecting specific areas of the genome.

I am currently working on nonhuman Ig. I may update with more details later when I work with human Ig.

The basic use only requires the short reads aligned BAM file:

```bash
IgCaller -I /path/to/IgCaller/IgCaller_reference_files/ -V hg19 -C ensembl -T /path/to/bams/tumor.bam -N /path/to/bams/normal.bam -R /path/to/reference/genome_hg19.fa -o /path/to/IgCaller/outputs/
```


- Output: IgCaller returns a set of tab-separated files:
    - tumor_sample_output_filtered.tsv: High confidence rearrangements passing the defined filters.
    - tumor_sample_output_IGH.tsv: File containing all IGH rearrangements.
    - tumor_sample_output_IGK.tsv: File containing all IGK rearrangements.
    - tumor_sample_output_IGL.tsv: File containing all IGL rearrangements.
    - tumor_sample_output_class_switch.tsv: File containing all CSR rearrangements.
    - tumor_sample_output_oncogenic_IG_rearrangements.tsv: File containing all oncogenic IG rearrangements (translocations, deletions, inversions, and gains) identified genome-wide.


More details:
Nadeu, F., Mas-de-les-Valls, R., Navarro, A. et al. IgCaller for reconstructing immunoglobulin gene rearrangements and oncogenic translocations from whole-genome sequencing in lymphoid neoplasms. Nature Communications 11, 3390 (2020). https://doi.org/10.1038/s41467-020-17095-7.





<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
