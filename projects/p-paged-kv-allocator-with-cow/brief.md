Our inference server is repeatedly hitting OutOfMemory (OOM) errors and crashing during beam search and parallel sampling workloads.

Currently, our sequence generation relies on a naive KV cache allocator that duplicates the entire KV history for every sequence generated from the same prompt. When branching generates 5-10 sequences (beams) from a long prompt, the system allocates thousands of redundant blocks, severely fragmenting the pool and exhausting physical memory almost instantly.

You must design and implement a Paged KV Allocator that natively supports block sharing and Copy-on-Write (CoW). The allocator should manage a pool of physical blocks (using a free list) and maintain logical block tables for each sequence.

When a sequence branches, the child should simply copy the parent's block table and increment the reference counts of the shared blocks. Memory should only be physically duplicated when a sequence tries to append a new token into a shared block that isn't fully filled (Copy-on-Write). The system must be robust, maintaining correct reference counts and avoiding memory leaks over long-running continuous batching traces.
