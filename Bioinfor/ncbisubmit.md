---
toc: true
url: ncbisubmit
covercopy: © Karobben
priority: 10000
date: 2024-10-17 23:36:10
title: "NCBI Data Submit with FTP/ASCP"
ytitle: "NCBI Data Submit with FTP/ASCP"
description: "NCBI Data Submit"
excerpt: "ASCP (Aspera Secure Copy Protocol) is a fast, reliable protocol for transferring large files, particularly over long distances or in conditions with network latency or packet loss. It uses a technology called fasp (Fast, Adaptive, and Secure Protocol) to maximize available bandwidth, making transfers faster than traditional methods like FTP.<br>For uploading data to NCBI, ASCP is particularly useful because it efficiently handles large datasets, such as genomic sequences or omics data. Its ability to resume interrupted transfers ensures that if a connection fails during an upload, the transfer continues from where it left off, saving time and bandwidth. ASCP also provides strong encryption, ensuring data security during the upload process."
tags: [Biology, Bioinformatics, Database] 
category: [Biology, Bioinformatics, Database]
cover: "https://imgur.com/7y0ZUbK.png"
thumbnail: "https://imgur.com/7y0ZUbK.png"
---


This post only talks about how to use the `ascp` to upload your sequencing data into NCBI.  

## 3 Different Ways of Submit Your Data

- Cloud from **Amazon S3** or **Google Cloud**
    If your data was stored in Amazon/Google Cloud at the beginning, you can easily and safely transfer them into NCBI. (I think so though I've never tried).
- **FTP** or **ASCP**
    I would recommend the second approach since our data was mostly stored in a Linux server. FTP and ASCP are very reliable. Especially for `ftp`, it is a very popular protocol. You could find a bunch of software like '[FileZilla](https://filezilla-project.org/)' to upload through ftp. The best feature of 'FileZilla' is it supports **resume interrupted transfer** and lists the fail-up loaded files so you can upload them again with one click. So, no matter how many and how big the files are, it can help you upload them safely. The main **limitation** for **FileZilla** is that it can't used in command form and so, is not suitable for the server.
- **Web Browser**
    Unless your data are very small, you would never want to try uploading them online.

## When to Upload Your Data

You can upload your data whenever you want. It is better to upload your file before you start to fill the submission tables. In step 7, you could find the data/director you submitted and include them in the submission. But it seems like the NCBI would delete an inactivated file within 30 days. It is long enough to finish the submission.

I prefer to use `FileZilla` to upload my data. But since now, all of my data are on the server and I don't want to download them again, I give the `ftp` and use `ascp`.

## How to use the ASCP

![](https://imgur.com/Wik8zeG.png)

ASCP is very easy and works similarly to `scp`. In the Submission home page, select ==My submissions== → ==Upload via Aspera command line or FTP== → ==Aspera command line instructions== to find the instructions. It would give you the download link and key for connecting to the NCBI server. After that, use it just like the `scp`.

Shortcomings or suggestions for `ascp`
1. **Keep everything in 1 director**: You'd like to upload all files into one directory. Because during the submission step **7 FILES**, you could only select 1 directory.
2. **No sub-directories**: You may want to upload the directory without subdirectories. Because in the submission portal, you can't check files in subdirectories. So, it is hard to track back which filed upload failed.
5. **Keep ascp log**: In the submission portal, you got only first 5 lines of uploaded files. So, remember to keep the `ascp` log to record the fail uploaded data. 
4. ==**upload the data one by one**==: Strongly recommend to upload each `fq` or compressed file one by one with scripts. When you try to upload the entire directory, it may fail (I never get it done when upload a directory)
5. After you upload your data, you can't see them until you go to step **7 FILES** in the submission portal.
6. You can't check them immediately even from the submission portal. It takes time for them to show in the **Step 7**

==The good thing is it is easy to write a script to upload your data automatically.==


!!! note In the instructions, it suggest you to use those parameters: `-QT -l100m -k1`
    1. **`-Q`**: This option disables the real-time display of progress and transfer statistics during the transfer. Normally, ASCP displays ongoing statistics, such as speed and percentage of completion, but using `-Q` will suppress this output.
    2. **`-T`**: This option disables encryption of the data stream during transfer. ASCP by default uses encryption for data security, but `-T` turns this off, which might improve transfer speed but at the cost of security.
    3. **`-l100m`**: This sets the transfer speed limit to **100 megabits per second**. You can adjust the value (e.g., `100m`) to control how fast the transfer is allowed to go, helping to prevent network congestion or manage bandwidth usage.
    4. **`-k1`**: This option controls file resume behavior. The value `1` means that if a transfer is interrupted, ASCP will resume from the point where it left off (resumable transfer). The other possible values for `-k` are:
       - `0`: No resume. The transfer restarts from the beginning.
       - `2`: Sparse resume. ASCP resumes only the missing parts of the file.

## Personal Experience

### Upload your data into a specific directory

Though we gave the argument `-k1`, it could still fail. In the log, it says:
<pre>
Partial Completion: 19711732K bytes transferred in 3695 seconds
(43691K bits/sec), in 6 files, 4 directories; 3 files failed.

Session Stop  (Error: Disk write failed (server))
</pre>
After you went to the step 7, you could see:
![](https://imgur.com/xxbYXVE.png)

Which means 2 directories are empty. In this case, you don't need to worry too much. You can change the code a little bit and continue to upload could solve this problem.
For example, you uploaded a directory named `ALL_RNA` with code `ascp -i $key_file -QT -l100m -k1 -d ALL_RNA $AddressFromInstruction`, the data in the directory `ALL_RNA/SAMPLEX` was failed to upload, you can use the code `ascp -i $key_file -QT -l100m -k1 -d ALL_RNA/SAMPLEX $AddressFromInstruction/ALL_RNA` to continue upload the directory `SAMPLEX` into the `ALL_RNA` in NCBI server
<pre> 
ascp -i $key_file -QT -l100m -k1 -d ALL_RNA $AddressFromInstruction
ascp -i $key_file -QT -l100m -k1 -d ALL_RNA<font color=red>/SAMPLEX</font> $AddressFromInstruction<font color=red>/ALL_RNA</font>
</pre>

### Upload your date in the script

When you have lots of data, one of the convenient ways I found is we could `ascp` each data independently. With a for loop, we could generate all codes into a script. When there are failed uploads, we just need to copy and paste the failed codes and run them again. Or we could also delete the code for the successfully uploaded one and run the entire script again.

```bash
for i in $(find YourDirectories -name ".fastq.gz"); do
    echo ascp -i $key_file -QT -l100m -k1 -d $i $AddressFromInstruction
done >> ascp.sh

bash ascp.sh
```

<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
