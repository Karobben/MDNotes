---
toc: true
url: MUMmer
covercopy: © Karobben
priority: 10000
date: 2024-06-21 18:29:31
title: "MUMmer: Rapidly Genomes Alignment"
ytitle: "MUMmer: Rapidly Genomes Alignment"
description: "MUMmer is a system for rapidly aligning entire genomes"
excerpt: "MUMmer is a system for rapidly aligning entire genomes. The current version (release 4.x) can find all 20 base pair maximal exact matches between two bacterial genomes of ~5 million base pairs each in 20 seconds, using 90 MB of memory, on a typical 1.8 GHz Linux desktop computer."
tags: [WGS, Genome, Bioinformatics, NGS]
category: [Biology, Bioinformatics, Software ]
cover: "https://imgur.com/D5P0f1g.png"
thumbnail: "https://imgur.com/D5P0f1g.png"
---


Installation:

In linux, you could simply install it by apt install mummer. The version of the mummer is around 3.1. The version from Bioconda is little bit new than the apt source but still kind of old. If you install teh MUMmer from those 2 source, you'll meet error when the reads is too long. 

## NUCmer:

Test target: Chicken Genome (GRCg6a) and Duck Genome (SKLA1.0)
It could be RAM monster when you compare 2 Genome directly. Especially when you want to use the multiple threads, the RAM would be occupied very quick and the program would be killed. 


```bash
nucmer -maxmatch -c 100 -p output_prefix reference.fasta query.fasta
show-coords -rcl output_prefix.delta > output_prefix.coords
cat output_prefix.coords
grep -Ev "^$|=|\[" result/test.coords| grep "^ "| awk 'BEGIN { OFS="\t"; print "Ref_Start", "Ref_End", "Query_Start", "Query_End", "Length1", "Length2", "Identity", "Ref", "Query" } {OFS="\t";  print $1, $2, $4, $5, $7, $8, $10, $12, $13}' > test.coords

```

But the problem is there are no such parameter for NUCmer to limited the use of RAM. The only way to solving this problem is split the genome into single sequences and processing one by one. 

<pre>
[1]    70706 killed     nucmer --maxmatch -c 100 -p result/Chicken-SKLA1.0
</pre>

![btop](https://imgur.com/iXM6Kr2.png)


I was tried to extract the first chromosome (196,202,544 bp, 192M) from the Chicken align against the Duck genome (1.2 Gb) and it takes 41 GB RAM if you given only 1 thread. If you run it without given threads, it would run with 3 threads and the RAM would increased into 70 GB. So, it seams it is save to run with about 7 threads with this size of data. But after 1h 12min, it was killed because of the increased demand of RAM. If you use single thread, the RAM would increased in to 77 after about 1h and 12min. Finally, it takes 13h 3m.

### MUMmer 

MUMmer is focusing on the difference between the reference and the Subject. It output is very sample. It only contains the start, end, and the length of the reference. Based on this, we could know that they are forward or reverse-complemented. It doesn't has the responded position for Subject chromosome. So, because of the low dimension information, it requires very low RAM and works very efficient. It could finishing calculation in a very short time.


```bash
mummer -mum -b -c ref.fa sub.fa  > ref.mums
mummerplot --postscript --prefix=ref_qry ref.mums
gnuplot ref_qry.gp
```


For visualization, you could use the package provide tools. You could also convert it into tables and visualize it with your favorite tools. So, after convert the mummer result into `tsv` by a python script, we could visualize the result with ggplot.

![mummer plots](https://imgur.com/KcnMXqg.png)


<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
