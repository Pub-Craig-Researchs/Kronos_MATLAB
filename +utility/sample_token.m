function token_dl = sample_token(logits, temperature, top_k, top_p)
%   SAMPLE_TOKEN vectorized sampling with Top-K and Top-P filtering
%
%   temperature: Scalar
%   top_k: Scalar (0 to disable)
%   top_p: Scalar (1.0 to disable)

arguments (Input)
    logits
    temperature (1,1) double = 1
    top_k (1,1) double = 0
    top_p (1,1) double = 0.9
end

arguments (Output)
    token_dl
end

raw = extractdata(logits);
if ndims(raw) == 3
    % Assuming TBC slice (1, Batch, Vocab) -> squeeze -> (Batch, Vocab)
    raw = squeeze(raw);
end
if size(raw,1)~=size(logits,3)
    raw = raw';
end

[Vocab, Batch] = size(raw);

% Apply Temperature
raw = raw ./ temperature;

% Strategy: We sort ONCE, compute masks ONCE, applied to entire batch.

[sorted_logits, sorted_indices] = sort(raw, 1, 'descend');

sorted_probs = softmax(dlarray(sorted_logits, "CB"));
sorted_probs = extractdata(sorted_probs);
cum_probs = cumsum(sorted_probs, 1);

% MASK GENERATION
filter_mask = false(Vocab, Batch);

% Top-K Mask
if top_k > 0 && top_k < Vocab
    % Everything after index K is filtered
    filter_mask(top_k+1:end,:) = true;
end

% Top-P Mask
if top_p < 1.0
    % Find cutoff: where cum_probs > top_p
    % Shift right to include the first token crossing threshold
    % mask = [false, (cum_probs > top_p)[:, :-1]]
    p_mask = cum_probs > top_p;
    p_mask(2:end) = p_mask(1:end-1);
    p_mask(1) = false; % Always keep top 1

    filter_mask = filter_mask | p_mask;
end

if any(filter_mask(:))
    sorted_logits(filter_mask) = -inf;
end

final_probs = softmax(dlarray(sorted_logits, "CB"));
final_probs = extractdata(final_probs); % (Batch, Vocab)

% Cumulative sum for sampling
cdf = cumsum(final_probs, 1);

r = rand(1,Batch);

sample_idx_in_sorted = sum(cdf < r, 1) + 1;

% Linear indexing:
linear_idx = sub2ind([Vocab,Batch],sample_idx_in_sorted, 1:Batch);
real_token_idx = sorted_indices(linear_idx);

% Convert to 0-based
idx = real_token_idx - 1;

% Output dlarray TB
token_dl = dlarray(idx, "TB");
end
