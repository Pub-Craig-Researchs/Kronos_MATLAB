function [mean_arr,std_arr,recon_data,stack_matrix] = kronos_inference(encoder, decoder, predictor, s2_decoder, input_data, timestamps, pred_len, options)
% KRONOS_INFERENCE Run autoregressive inference with Top-K/Top-P Sampling
%
%   Inputs:
%       input_data: dlarray [Batch, Seq, Feat] ("BTC")
%       timestamps: dlarray [TotalSeq, Batch, 5] ("TBC")
%       pred_len:   Integer
%       options:    Struct with fields:
%                   - Temperature (default 1.0)
%                   - TopK (default 0, disabled)
%                   - TopP (default 0.9, Nucleus Sampling)
%                   - ifPrint (default false)
%
%   Outputs:
%       predictions: Predicted data
%       recon_data:  Reconstructed history
input_data = gpuArray(input_data);
timestamps = gpuArray(timestamps);

if nargin < 8
    options = struct();
end
temp = 1.0;
if isfield(options, 'Temperature'), temp = options.Temperature; end

top_k = 0;
if isfield(options, 'TopK'), top_k = options.TopK; end

top_p = 0.9;
if isfield(options, 'TopP'), top_p = options.TopP; end

ifprint = false;
if isfield(options, 'ifPrint'), ifprint = options.ifPrint; end

rounds = 1;
if isfield(options, 'Rounds'), rounds = options.Rounds; end

% Encode
[s1, s2] = predict(encoder, input_data);


% Autoregressive Loop
if(ifprint)
    fprintf('Starting inference (T=%.1f, TopK=%d, TopP=%.2f, Rounds=%.1f) for %d steps...\n', temp, top_k, top_p, rounds, pred_len);
end

predictions = cell(1,rounds);
for round = 1:rounds
    curr_s1 = dlarray(s1,"TB");
    curr_s2 = dlarray(s2,"TB");
    for i = 1:pred_len % i=2
        current_seq_len = size(curr_s1, 2);

        if size(timestamps, 3) < current_seq_len
            error('Timestamp length insufficient.');
        end
        curr_stamp = timestamps(:, :, 1:current_seq_len);

        % Predictor
        [s1_logits_all, ~, context_all] = predict(predictor, curr_s1, curr_s2, curr_stamp);

        s1_logits_last = s1_logits_all(end, :, :);
        context_last   = context_all(end, :, :);

        % Sample S1
        next_s1_dl = utility.sample_token(s1_logits_last, temp, top_k, top_p);

        % S2 Decoder
        s2_logits_out = predict(s2_decoder, context_last, next_s1_dl);

        % Sample S2
        next_s2_dl = utility.sample_token(s2_logits_out, temp, top_k, top_p);

        % Append
        curr_s1 = cat(2, curr_s1, next_s1_dl);
        curr_s2 = cat(2, curr_s2, next_s2_dl);

        if(ifprint)
            if mod(i, 10) == 0 || i == pred_len
                fprintf('Generated step %d/%d\n', i, pred_len);
            end
        end
    end

    % Decode
    full_output = predict(decoder, curr_s1, curr_s2);

    full_data = extractdata(full_output);
    input_len = size(input_data, 3);

    recon_data  = gather(full_data(1:input_len, :, :));
    predictions{round} = gather(full_data(input_len+1:end, :, :));
end
stack_matrix = cat(4, predictions{:});

mean_arr = mean(stack_matrix, 4);
std_arr = std(stack_matrix, 0, 4);
end


