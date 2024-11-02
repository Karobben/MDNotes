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
tags: [Bioinformatics, Biochmistry, Biology, 3D, Plot]
category: [Biology, Bioinformatics, Protein Structure]
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

```python
# codes for set a pure white background
# Those codes only working on the first few steps.
# I didn't figure how to use script to make the Geometry Nodes 
# So, After you run those 3 commands, you still need to started from step 3 in the second pictures to manually finish the Geometry Nodes setting.
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.space_data.shading.type = 'RENDERED'

bpy.context.object.scale[0] = 3.5
bpy.context.object.scale[1] = 3.5
bpy.context.object.scale[2] = 3.5


bpy.context.scene.render.film_transparent = True
bpy.context.scene.view_settings.view_transform = 'Standard'
bpy.context.scene.use_nodes = True
```

## Different Colors in a Surface 

|||
|:-|:-|
|![Settings in Geomitry](https://imgur.com/tTOeKDl.png)| The key idea for given different color is by rendern multiple layers of color on the surface. By reverse select residues , we could delete the colors from selected layer and expose the color from inner layer.| 
|![Results](https://imgur.com/MGA9Mqk.png)| Final resutls show|

## Multiple Style in One Object

|![Blender Join Geometry](https://imgur.com/TVEMkDe.png)||
|:-|:-|
|![](https://imgur.com/AOUUkWK.png)| Like the example in the picture, it rendered both surface model and the stick model in one object. This is achieved by `Join Geometry` |

## Customize the Color From The Surface

|![](https://imgur.com/QU1xa7K.png)|![](https://imgur.com/7rdCxfl.png)|
|:-:|:-:|

For Customizing the surface color, there are 2 ways to do it.

1. using `pLDDT` nodes from `Color`
2. using the `Color Attribute Map` nodes from `Color`.

In both case, they are actually using the same set of value stored in `pdb` or `cif` file.
In the pdb format show below, the 11th column marked as white is the value for `pLDDT`. If you want to manage it with `Color Attribute Map`, the name of it is `b_factor`

<pre>
<font size=1>ATOM   4365  C   ASP C 150      17.854  27.766  83.090  1.00 <font color=white>99.42</font>      C    C  
ATOM   4366  O   ASP C 150      17.369  28.239  82.038  1.00 <font color=white>95.32</font>      C    O  
ATOM   4367  CB  ASP C 150      19.712  26.091  82.521  1.00 <font color=white>98.18</font>      C    C  
ATOM   4368  CG  ASP C 150      20.447  24.817  82.987  1.00 <font color=white>96.59</font>      C    C  
ATOM   4369  OD1 ASP C 150      20.121  24.255  84.056  1.00 <font color=white>96.78</font>      C    O  
ATOM   4370  OD2 ASP C 150      21.402  24.406  82.277  1.00 <font color=white>96.06</font>      C    O1-
ATOM   4371  OXT ASP C 150      18.041  28.393  84.184  1.00 <font color=white>95.18</font>      C    O1-
</font></pre>




## Watching List

- [ ] [Select color pallet](https://www.youtube.com/watch?v=sIblmWV0NuM)


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




