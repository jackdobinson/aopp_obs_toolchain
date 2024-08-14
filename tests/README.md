

## Purpose of Tests ##

Software tests exist for a few reasons:

1) To ensure that each component (function, class, method, etc.) of some software does the task it is purported to do. 

2) To act as a *canary*, i.e. to ensure that changes to the code-base does not break any documented functionality of the software.

3) To enhance security by finding and guarding against known vulnerabilities.

4) To provide a way to consistently measure software performance.

From the point of view of *aopp_obs_toolchain* (which is an evolving codebase) (2) is probably the most important, followed by (1). We want to be able to iterate upon the software and fix errors that are introduced by each iteration quickly. With that in mind, we should concentrate on testing larger chunks of functionality. 

For example:

* Don't test:
  - Exactly how a file is read into memory, only that is is read into memory correctly. 
  - How an image is centered, only that it is centered.
  - The individual steps of a process, only the end result of the process.
* Do test:
  - Large functions/methods complete and perform their tasks correctly.
  - Incompatible inputs to processes are detected reliably.
  - End-to-end correct completion of an entire process.