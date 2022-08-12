# Current Bladebit Disk Beta TODOS

- [x] Fix compiler issues that popped up on macOS/clang after eliminating wrnings on Linux & Windows.
- [] Fix 128 buckets bug (linepoints are not serialized incorrectly)
- [x] Fix 1024 buckets bug (crashes on P1 T4, probably related to below large block crash)
- [] Fix 1024 buckets bug on P3
- [] Fix crash on P1 T4 w/ RAID, which actually seems to be w/ big block sizes. (I believe this is due to not rounding up some buffers to block sizes. There's anote about this in fp code.)
- [x] Add no-direct-io flag for both tmp dirs
- [x] Add no-direct-io flag for final plot
- [] Perhaps add a different queue for t2 if it's a different physical disk
- [-] Add k32 bounded/non-overflowing version
- [] Add synchronos tmp IO (good with cache)
- [x] Add interleaved-only writing method for bigger write chunks/more sequential I/O.
 - [] Integrate this into other phases
- [x] Add method to reduce cache requirements to 96G instead of 192G
 - [] Integrate cache reduction into the plotting process
- [] Bring in avx256 linepoint conversion (already implemented in an old BB branch)
- [] Allow sub temp directories or plot-speific file temp file names (allows for concurrent plotting).
