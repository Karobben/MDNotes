---
toc: true
url: blender-molecular-nodes
covercopy: © Karobben
priority: 10000
date: 2024-10-19 09:53:38
title: "Render Your Protein in Blender with Molecular Nodes"
ytitle: "Render Your Protein in Blender with Molecular Nodes"
description: "Render Your Protein in Blender with Molecular Nodes"
excerpt: "Render Your Protein in Blender with Molecular Nodes"
tags: []
category: []
cover: "https://imgur.com/RZ6Pv2O.png"
thumbnail: "https://imgur.com/RZ6Pv2O.png"
---

## Who to Install Molecular Nodes for Blender

First, you should download Blender yourself. Instead of the latest version, opt for a stable version because the newest release may have bugs or be incompatible with Molecular Nodes. I tried version 4.40, but when I changed the style of the molecule to Ribbon or another style, Blender crashed and closed itself. Then, I switched to version 4.2.2, and it worked fine.

As following the figure "Install the Extension", you can find this plugin and install it. Once you down installation, you can find there is some thing new pops up like its show in figure "Update in Scene". In this new module, you could download the pdb online or when you have pdb in the "Cache Downloads" directory, you could also load it with "Molecular Nodes". When you load the molecular, it looks terrible. You need to follow the figure "Render By Cycles" and "Start Render" to get a normal view of molecular("Atoms View").

![Install the Extension](https://imgur.com/uCbxiP9.png) 

![Update in Scene](https://imgur.com/9fEK7wf.png)

![Render By Cycles](https://imgur.com/ehHyKKP.png)
![Start Render](https://imgur.com/1DMegwb.png)
![Atoms View](https://imgur.com/LoFewhU.png)

## Add a Pure Perfect Background

Source: [YouTube: EMPossible](https://www.youtube.com/watch?v=aegiN7XeLow)

![Pure White Background](https://imgur.com/Wl0ea71.png)


How to set:

| | |
| :---: | :-- |
| ![Set pure background](https://imgur.com/UndFMUw.png) | select <li>"Render" → "Film" → "Transparent"<li>"Render" → "Color Management" → "View Transform" → "Starndard" | 
| ![Set Compositing](https://imgur.com/jxy0bGJ.png)| Set the Compositing. And that's it. Go to rendering and it woud add an Perfectwhite at the background|


## Different Colors in a Surface 

|||
|:-|:-|
|![Settings in Geomitry](https://imgur.com/tTOeKDl.png)| The key idea for given different color is by rendern multiple layers of color on the surface. By reverse select residues , we could delete the colors from selected layer and expose the color from inner layer.| 
|![Results](https://imgur.com/MGA9Mqk.png)| Final resutls show|


## Trouble Shoot

### Dead Black in Transparent

| Dead Black    | Change Setting | After Change    |
|---------------- | --------------- | --------------- |
|![Befor Change](https://imgur.com/jCo5bO3.png)  | ![Change Setting](https://imgur.com/IaT6UlB.png) | ![After Correction](https://imgur.com/Zb3XWZz.png)    |



<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
