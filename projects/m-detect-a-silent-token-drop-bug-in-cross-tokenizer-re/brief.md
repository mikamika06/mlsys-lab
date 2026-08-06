# Ticket: Silent Token-Drop Bug in Cross-Tokenizer Realignment for Speculative Decoding

During cross-tokenizer speculative decoding executions between draft and target models, we have been encountering sporadic sequence generation stalls and severe downstream token drift. When the draft model proposes a sequence of candidate tokens, the system performs a cross-tokenizer realignment pass to map the tokens into the target tokenizer's vocabulary space.

Under certain multi-token chunk boundaries and punctuation-heavy sequences, an insidious bug occurs: a token from the draft sequence is silently dropped or omitted during the re-encoding mapping step without throwing any exception, assertion error, or warning. Because the token count shrinks silently, the corresponding position IDs and attention masks fall out of alignment. Consequently, the target model processes corrupted context windows, resulting in degraded output quality, infinite repetition loops, or sudden premature termination.

We need to implement a cross-tokenizer realignment and validation pipeline that accurately tracks token mapping trajectories, computes byte-exact matching fractions across realigned streams, and includes a comprehensive regression test suite to catch any silent token dropping behavior during alignment updates.
