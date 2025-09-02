---
title: "tmux2"
description: "tmux2"
url: tmux2
date: 2020/06/26
toc: true
excerpt: "Termux Termux is an Android terminal emulator and Linux environment app that works directly with no rooting or setup required. A minimal base system is installed automatically - additional packages are available using the APT package manager. Read the wiki to learn more..."
tags: [Linux, CLI Tools]
category: [Linux, others]
cover: 'https://s1.ax1x.com/2020/06/26/NsmGxf.png'
thumbnail: 'https://s1.ax1x.com/2020/06/26/NsmGxf.png'
priority: 10000
---

## Genearl Commands for Tmux 

![NsmGxf.png](https://s1.ax1x.com/2020/06/26/NsmGxf.png)

tmux is a very powerful interact-able bash interpreter. Once you familiarized with the hot keys, it would be an inextricable programs of you.

Copy and Past:

| Moves    | Keys     |
| :------------- | :------------- |
| Enter Copy Mode | `ctrl`-`b` + `[` |
| Start to selecte| `ctrl`-`b` + `blank` |
| Copy the selected words| `ctrl`-`b` + `w` |
| Copy the selected words| `Enter` |
| Paste the words| `ctrl`-`b` + `]` |
| Show the copy list| `ctrl`-`b` + `#` |
| Select the copy list| `ctrl`-`b` + `=` |
| Jump to the line head| `ctrl`-`a` |
| Jump to the line end| `ctrl`-`e` |




Panes: Each window could split into small panes which is the key feature for tmux.

| Moves    | Keys     |
| :------------- | :------------- |
| Split pane horizontal | `ctrl`-`b` + `"` |
| Split pane Vertical| `ctrl`-`b` + `%` |
| Resize the the Panes | `ctrl`-`b` + `ctrl`-`→` |
| Show pane ID| `ctrl`-`b` + `q`|
| Show the window and pane ID| `ctrl`-`b` + `w`|
| move the Panes into right| `ctrl`-`b` + `{ `|


Tips:
1. Panes resize:
    After you executed `ctrl`-`b`, you can hold `ctrl` and press the up, down, left, or right as many as you can until it fits you the best.

## Window Related

| Moves| Keys|
|:-----|:-----|
| Create a new Window|`ctrl`-`b`+ 'c'|
| Switch to the next window| `ctrl`-`b` + `n`|
| Move the window eadge to the right| `ctrl`-`b` + `→`|
| flip the window lift| `ctrl`-`b` + `{ `|
| flip the window right| `ctrl`-`b` + `}`|


## Detach and Attach

In tmux, you can detach the current session and attach it later. It is very useful when you are working on a remote server and you need to leave for a while. Basically, the detach the session means you can leave the session and the session will keep running in the background. And then, you can savly log out the server. When you come back, you can attach the session and continue your work.

| Moves| Keys|
|:-----|:-----|
| Detach the session| `ctrl`-`b` + `d`|
| List the session| `tmux list-sessions`|
| Attach the session| `tmux attach-session -t <session-number>`|

This could be very helpful.
Example:
```bash
tmux list-sessions
tmux attach-session -t 10
```

List of the session will be like:
<pre>
10: 1 windows (created Tue Jan 28 11:01:30 2025)
11: 1 windows (created Tue Jan 28 11:05:37 2025)
12: 1 windows (created Tue Jan 28 11:06:43 2025)
20: 1 windows (created Fri Jan 31 10:11:23 2025) (attached)
21: 1 windows (created Fri Jan 31 10:21:27 2025)
9: 1 windows (created Tue Jan 28 11:01:14 2025)
</pre>
