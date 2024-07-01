---
toc: true
url: juicer
covercopy: © Karobben
priority: 10000
date: 2024-06-27 15:55:58
title: "Juicer: a One-Click System for Analyzing Loop-Resolution Hi-C Experiments"
ytitle: "Juicer: a One-Click System for Analyzing Loop-Resolution Hi-C Experiments"
description: "Juicer: a One-Click System for Analyzing Loop-Resolution Hi-C Experiments"
excerpt: "Hi-C experiments explore the 3D structure of the genome, generating terabases of data to create high-resolution contact maps. Here, we introduce Juicer, an open-source tool for analyzing terabase-scale Hi-C datasets. Juicer allows users without a computational background to transform raw sequence data into normalized contact maps with one click. Juicer produces a hic file containing compressed contact matrices at many resolutions, facilitating visualization and analysis at multiple scales. Structural features, such as loops and domains, are automatically annotated."
tags:
category:
cover: "https://www.cell.com/cms/attachment/4173c5c1-9206-4243-a997-03ecf630da5d/gr1.jpg"
thumbnail: "https://www.cell.com/cms/attachment/4173c5c1-9206-4243-a997-03ecf630da5d/gr1.jpg"
---

Prerequisite :

```bash
conda install bwa               # for short reads alignment
conda install samtools          # for reading the align results 

```

Resources:
- Paper: Durand NC, Shamim MS, Machol I, Rao SSP, Huntley MH, Lander ES, et al. Juicer Provides a One-Click System for Analyzing Loop-Resolution Hi-C Experiments. Cell Syst. 2016;3:95–8.
- GitHub Source code: [aidenlab/juicer](https://github.com/aidenlab/juicer)
- Example Pipeline: [ENCODE-DCC/hic-pipeline](https://github.com/ENCODE-DCC/hic-pipeline)
- Forums: [3d-genomics; google](https://groups.google.com/g/3d-genomics). They suggest to talk and ask on the google group rather than the github issue because you could got faster responds there.


For working on hic-pipeline, if you want to run it in local machine, make sure that `docker` is installed. I don't have docker installed, so, I'll giving this try up.


When you got the mistake and want run again, make sure remove those directories first.

```bash
rm -rf /home/wenkanl2/Tomas/20240614_DuckGenome/myJuicerDir/aligned

```



```bash
# HiCCUPS
scripts/common/juicer_tools hiccups --ignore_sparsity aligned/inter_30.hic aligned/inter_30_loops
# APA: 
scripts/common/juicer_tools apa aligned/inter_30.hic aligned/inter_30_loops apa_results
```


In the test data, it generally takes 90GB RAM and 7 GB of GPU RAM

## Troubleshooting

<pre>
HiCCUPS:

GPUs are not installed so HiCCUPs cannot be run

(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:
***! Error! Either inter.hic or inter_30.hic were not created
Either inter.hic or inter_30.hic were not created.  Check  for results
</pre>

Check if cuda is installed appropriate. If so, Check if it is in your working environment.

How to add it in your working environment: (edit your `~/.bashrc` or `~/.zshrc` file)

```~/.bashrc
export PATH="/usr/local/cuda-8.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"
```

Or you could add the '--cpu' flag on file `scripts/common/juicer_postprocessing.sh`

```diff
- if hash nvcc 2>/dev/null
- then
-    ${juicer_tools_path} hiccups ${hic_file_path} ${hic_file_path%.*}"_loops"
+    ${juicer_tools_path} hiccups --cpu ${hic_file_path} ${hic_file_path%.*}"_loops"
-    if [ $? -ne 0 ]; then
-	echo "***! Problem while running HiCCUPS";
-	exit 1
-    fi
-else
-    echo "GPUs are not installed so HiCCUPs cannot be run";
-fi
```

When the `if hash nvcc 2>/dev/null` detected that the nvcc doesn't in the environment, it would exit. So, you may like to delete the entail if statement. 


<pre>
HiCCUPS:

Picked up _JAVA_OPTIONS: -Xmx150000m -Xms150000m
Reading file: /home/wenkanl2/Tomas/20240614_DuckGenome/myJuicerDir/aligned/inter_30.hic
No valid configurations specified, using default settings
Warning Hi-C map is too sparse to find many loops via HiCCUPS.
Exiting. To disable sparsity check, use the --ignore_sparsity flag.
</pre>

As it suggests, you need to add the `--ignore_sparsity` flag. But, again, you can only make this change by alter `scripts/common/juicer_postprocessing.sh` 

```diff
if hash nvcc 2>/dev/null
then
-    ${juicer_tools_path} hiccups ${hic_file_path} ${hic_file_path%.*}"_loops"
+    ${juicer_tools_path} hiccups  --ignore_sparsity ${hic_file_path} ${hic_file_path%.*}"_loops"
    if [ $? -ne 0 ]; then
	echo "***! Problem while running HiCCUPS";
	exit 1
    fi
else
    echo "GPUs are not installed so HiCCUPs cannot be run";
fi
```



0 loops

<pre>
100% 
0 loops written to file: ...
HiCCUPS complete
</pre>

According to this [post answer by Neva Durand](https://groups.google.com/g/3d-genomics/c/9f5UUhuS8O4/m/RTE1YVTKAgAJ), it could be the data is too sparse. 
> Yes, it’s an order of magnitude too few reads to find loops. You need to do deeper sequencing  / more replicates and then combine them. You need at least 1 billion reads. Otherwise your experiments simply don’t have the depth to determine loops (with any algorithm). 




<pre>
Not including fragment map
Error while reading graphs file: java.io.FileNotFoundException: /home/wenkanl2/Tomas/20240614_DuckGenome/myJuicerDir/aligned/inter_30_hists.m (No such file or directory)
Start preprocess
Writing header
Writing body
java.lang.RuntimeException: No reads in Hi-C contact matrices. This could be because the MAPQ filter is set too high (-q) or because all reads map to the same fragment.
	at juicebox.tools.utils.original.Preprocessor$MatrixZoomDataPP.mergeAndWriteBlocks(Preprocessor.java:1650)
	at juicebox.tools.utils.original.Preprocessor$MatrixZoomDataPP.access$000(Preprocessor.java:1419)
	at juicebox.tools.utils.original.Preprocessor.writeMatrix(Preprocessor.java:832)
	at juicebox.tools.utils.original.Preprocessor.writeBody(Preprocessor.java:582)
	at juicebox.tools.utils.original.Preprocessor.preprocess(Preprocessor.java:346)
	at juicebox.tools.clt.old.PreProcessing.run(PreProcessing.java:116)
	at juicebox.tools.HiCTools.main(HiCTools.java:96)

real	2m7.365s
user	0m33.647s
sys	0m49.450s
Picked up _JAVA_OPTIONS: -Xmx150000m -Xms150000m
Error reading datasetnull
java.io.EOFException
	at htsjdk.tribble.util.LittleEndianInputStream.readFully(LittleEndianInputStream.java:138)
	at htsjdk.tribble.util.LittleEndianInputStream.readLong(LittleEndianInputStream.java:80)
	at htsjdk.tribble.util.LittleEndianInputStream.readDouble(LittleEndianInputStream.java:100)
	at juicebox.data.DatasetReaderV2.readFooter(DatasetReaderV2.java:470)
	at juicebox.data.DatasetReaderV2.read(DatasetReaderV2.java:235)
	at juicebox.tools.utils.original.NormalizationVectorUpdater.updateHicFile(NormalizationVectorUpdater.java:78)
	at juicebox.tools.clt.old.AddNorm.run(AddNorm.java:84)
	at juicebox.tools.HiCTools.main(HiCTools.java:96)

real	0m0.706s
user	0m1.229s
sys	0m0.399s
/home/wenkanl2/Tomas/20240614_DuckGenome/myJuicerDir/scripts/common/juicer_postprocessing.sh: option requires an argument -- g
Usage: /home/wenkanl2/Tomas/20240614_DuckGenome/myJuicerDir/scripts/common/juicer_postprocessing.sh [-h] -j <juicer_tools_file_path> -i <hic_file_path> -m <bed_file_dir> -g <genome ID>
***! Error! Either inter.hic or inter_30.hic were not created
Either inter.hic or inter_30.hic were not created.  Check  for results
</pre>


<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
