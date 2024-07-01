---
toc: true
url: LongReads
covercopy: © Karobben
priority: 10000
date: 2024-06-30 15:50:52
title: "Nanopre and PacBio based Genome Assembly"
ytitle: "Nanopre and PacBio based Genome Assembly"
description: "Nanopre and PacBio based Genome Assembly" 
excerpt: "Nanopre and PacBio based Genome Assembly"
tags: []
category: []
cover: ""
thumbnail: ""
---

Related Papers:
- Evaluating Illumina-, Nanopore-, and PacBio-based genome assembly strategies with the bald notothen, Trematomus borchgrevinki; [Paper](https://academic.oup.com/g3journal/article/12/11/jkac192/6651842)
- 

## Evaluating Illumina-, Nanopore-, and PacBio-based genome assembly strategies with the bald notothen, Trematomus borchgrevinki

In this paper, they did long reads assembly and Long-reads, short-reads hybrid assembly comparing. The experiment organism is "Trematomus borchgrevinki", a cold specialized Antarctic notothenioid fish with an estimated genome size of 1.28 Gb

**Hybrid** assemblies can generate ==higher contiguity== they tend to suffer from lower quality. **long-read-only assemblies** can be optimized for ==contiguity== by subsampling length-restricted raw reads. Long-read contig assembly is the current **best choice** and that assemblies from phase I and phase II were of lower quality.

Strategies:
- Long-reads and short-reads hybrid: quickmerge
    1. Long-reads assembly independently: `Canu` and `WTDBG2` assembly, assessed with `QUAST`
    2. 2 rounds of polishing with `Pilon`. (First round: SNPs adn indels, Second round: local reassembly)
    3. Gap filling with `PBJELLY`
- Long-reads only was assembly by variaties of tools. The yacrd (Marijon et al. 2020) it the tool to identify potential **chimeric reads**
    1. `WTDBG2` was used to do the assembly


For long-reads, comparing to short-reads assembled genome, it has high continuity but also more number of duplicated BUSCO genes. Chimeric reads are exist

<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
