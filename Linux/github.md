---
title: "github"
description: "github"
url: github2
date: 2020/06/23
toc: true
excerpt: "Basic commands for synchrony the local and cloud github repository."
tags: [Linux, Scripting, bash, CLI Tools, git]
category: [Linux, GitHub]
cover: 'https://kinsta.com/wp-content/uploads/2018/04/what-is-github-1-1-1024x512.png'
thumbnail: 'https://cn.bing.com/th?id=AMMS_c45f125b765170342ef8efd07cb7a55f&w=110&h=110'
priority: 10000
covercopy: <a href="https://kinsta.com/knowledgebase/what-is-github">© kinsta</a>
---

## github


Github 本地上更新库
```bash
## Initialize your directory
git init

## 关联
git remote add IO https://github.com/Karobben/Karobben.github.io

##添加
git add .

##注释
git commit -m "注释"

git pull --rebase IO master
git push -u IO master
```

## Avoid password everytime

[Click me](https://luanlengli.github.io/2019/04/07/git-pull%E5%85%8D%E5%AF%86%E7%A0%81%E9%85%8D%E7%BD%AE.html)

## when uploading file is to big

```bash
git config http.postBuffer 524288000
```

## ignore files

Cite: [Git-scm](https://git-scm.com/docs/gitignore)

the name of the file: `.gitignore` should keep in the home directory rather than the `.git` directory

```txt
$ cat .gitignore
# exclude everything except directory foo/bar
/*
!/foo
/foo/*
!/foo/bar
```

- An optional prefix "!" which negates the pattern; any matching file excluded by a previous pattern will become included again. It is not possible to re-include a file if a parent directory of that file is excluded.
- An asterisk "*" matches anything except a slash. 
- A trailing "/**" matches everything inside. For example, "abc/**" matches all files inside directory "abc", relative to the location of the .gitignore file, with infinite depth.
- A slash followed by two consecutive asterisks then a slash matches zero or more directories. For example, "a/**/b" matches "a/b", "a/x/b", "a/x/y/b" and so on.

## Delete large files

[© Daniel Andrei Mincă; 2015](https://stackoverflow.com/questions/33360043/git-error-need-to-remove-large-file)
```bash
git rm --cached giant_file
# Stage our giant file for removal, but leave it on disk

git commit --amend -CHEAD
# Amend the previous commit with your change
# Simply making a new commit won't work, as you need
# to remove the file from the unpushed history as well

git push
# Push our rewritten, smaller commit
``` 

Another way to solve the same problem

[© Clark McCauley; 2022](https://stackoverflow.com/questions/19573031/cant-push-to-github-because-of-large-file-which-i-already-deleted)
```bash
git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch <file/dir>' HEAD
```

## Git push with ssh

Documentation: [Github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh)

Follow the instructions from Github to generate a ssh public key first.
Be sure about add the email for config the user

Exp:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519
```

After that, copy the key into github.

You may still find that Username is needed for push.
According [2240](https://stackoverflow.com/questions/6565357/git-push-requires-username-and-password), we need to change the type or remote link.

### Test Your Connection

Once you down, you could test the ssh connections with `ssh -T git@github.com`. If it works fine, you'll get the greating form GitHub:

<pre>
Hi Karobben! You've successfully authenticated, but GitHub does not provide shell access.
</pre>

If you are running on another environment, you'll get a warning. But it would be fine, just input `yes` would solve all problems.
<pre>
Warning: the ECDSA host key for 'github.com' differs from the key for the IP address '140.82.114.3'
Offending key for IP in /home/ken/.ssh/known_hosts:9
Matching host key in /home/ken/.ssh/known_hosts:69
Are you sure you want to continue connecting (yes/no)?
</pre>

### Ready Your Local Repository

Enter your github repository page and select the ssh link to configure the local repository as follow and the problem shell be solved.

![github repository](https://s1.ax1x.com/2023/02/11/pSh4zNT.png)

```bash
git remote set-url origin git@github.com:username/repo.git
```

## Merge Conflict

When you have a merge conflict, you need to resolve it manually. Here's how you can resolve a merge conflict in a file.

1. Open the file in a text editor and look for the conflict markers. Conflict markers are added by Git to indicate the conflicting changes from different branches. They look like this:

<pre>
<<<<<<< HEAD
This is the content from the current branch
=======
This is the content from the branch you're merging in
>>>>>>> branch-name
</pre>

```bash
# Open the file in a text editor and resolve the conflict markers
lvim lazy-lock.json
# After resolving the conflicts, stage the file
git add lazy-lock.json
# Commit the merge
git commit -m "Resolved merge conflict in lazy-lock.json"
# If you were rebasing, continue the rebase
git rebase --continue
```

## Re-base the Local by Deleting all Local Change

```bahs
git stash
git pull
```

Reference: [Cameron McKenzie](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/remove-revert-discard-local-uncommitted-changes-Git-how-to)
- The git stash command, which saves uncommitted changes and reset your workspace.
- The git reset command, which only touches tracked files.
- The git clean command, that deletes every untracked file.

## Errors

### fatal: in unpopulated submodule '.deploy_git'

<pre>
fatal: in unpopulated submodule '.deploy_git'
FATAL { err:
   { Error: Spawn failed
       at ChildProcess.task.on.code (/mnt/8A26661926660713/Github/Notes_BK/node_modules/hexo-util/lib/spawn.js:51:21)
       at ChildProcess.emit (events.js:198:13)
       at Process.ChildProcess._handle.onexit (internal/child_process.js:248:12) code: 128 } } 'Something\'s wrong. Maybe you can find the solution here: %s' '\u001b[4mhttps://hexo.io/docs/troubleshooting.html\u001b[24m'
</pre>


> According to ChatGPT (this is a super genius! I can't find this result anywhere!) The error message you provided is related to Hexo, a static site generator. It seems that there was an issue with a submodule named '.deploy_git' in the Git repository you were working with.
>
> This error occurs when there's a problem with the Git submodule in your Hexo project, and it's not able to be cloned. To resolve this issue, try the following steps:

```bash
# Remove the problematic submodule:
git rm --cached .deploy_git
# Commit the changes:
git commit -m "Removed problematic submodule"
# Re-add the submodule:
# Replace <repo> with the URL of the Git repository for the submodule.
git submodule add -b master <repo> .deploy_git
# Initialize the submodule:
git submodule init
# Update the submodule:
git submodule update
```

> These steps should resolve the issue with the Git submodule and allow you to continue using Hexo to generate your static site. If the issue persists, you may need to refer to the Hexo troubleshooting guide (https://hexo.io/docs/troubleshooting.html) for further assistance.





<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>

## Submodule

In github, you could add entire another repository here with submodule:
```bash
git submodule add  git@github.com:Karobben/20240501_PacBio_lib_assmb_screening.git
```

How to remove the submodule?

To remove a submodule from a Git repository, you'll need to follow a series of steps to cleanly remove all traces of the submodule from your repository. Here's how to do it:

### 1. **Deinitialize the Submodule**
   First, you need to deinitialize the submodule to remove its configuration. Run the following command:

   ```bash
   git submodule deinit -f path/to/submodule
   ```

   Replace `path/to/submodule` with the actual path to the submodule.

### 2. **Remove the Submodule from the `.gitmodules` File**
   The `.gitmodules` file in the root of your repository contains the configuration for all submodules. You need to remove the corresponding entry of the submodule you want to delete.

   Open the `.gitmodules` file in a text editor and remove the section corresponding to the submodule. It will look something like this:

   ```ini
   [submodule "path/to/submodule"]
   	path = path/to/submodule
   	url = https://github.com/user/repo.git
   ```

   Save and close the file after removing the relevant section.

### 3. **Remove the Submodule Directory from the Working Tree**
   After removing the entry from `.gitmodules`, you can remove the submodule directory from your working tree:

   ```bash
   rm -rf path/to/submodule
   ```

### 4. **Remove the Submodule from the Git Index**
   Finally, you need to remove the submodule from the Git index (staging area). Run the following command:

   ```bash
   git rm -f path/to/submodule
   ```

   This command will remove the submodule from your repository's index, which means it will be removed in the next commit.

### 5. **Commit the Changes**
   Now that you've removed the submodule, commit the changes:

   ```bash
   git commit -m "Remove submodule path/to/submodule"
   ```

### 6. **Remove the Submodule Directory from the Git Directory**
   As an optional cleanup step, you can remove the submodule's entry in the `.git/config` file and the submodule directory inside the `.git` folder. The commands below accomplish this:

   ```bash
   rm -rf .git/modules/path/to/submodule
   ```

This sequence of commands will completely remove the submodule from your repository. If you push these changes to a remote repository, the submodule will also be removed from there.

## Branches

- Delete branch: `git push origin --delete <remote-branch-name>`

- Upload local repository into a new branch. ([source](https://stackoverflow.com/questions/2765421/how-do-i-push-a-new-local-branch-to-a-remote-git-repository-and-track-it-too))
    ```bash
    git checkout -b <branch>
    git push -u origin <branch>
    ```
