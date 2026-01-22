function [pred_close,pred_ret,predictions] = Kronos(model,Tbl,IN_SAMPLE_RANGE,PRED_LEN,future_times,Kronos_Options)
%   Tbl should be a table or time table. 
%   Varibales should be ["Time","open","high","low","close","volume","amount"].
%   If volume and amount are unavaliable use zeros(height(Tbl),1) to FILL.

arguments (Input)
    model (1,1) struct
    Tbl {mustBeA(Tbl, ["table", "timetable"])}
    IN_SAMPLE_RANGE (1,:) double
    PRED_LEN (1,1) double
    future_times datetime = NaT(1,1)
    Kronos_Options.Rounds (1,1) double = 1
    Kronos_Options.Seed (1,1) double = 0
    Kronos_Options.Temperature (1,1) double = 1
    Kronos_Options.TopK (1,1) double = 0
    Kronos_Options.TopP (1,1) double = 0.9
    Kronos_Options.ifPrint (1,1) double = false
    Kronos_Options.Type (1,1) string = "Prediction"
end

arguments (Output)
    pred_close double
    pred_ret double
    predictions (:,6,:)
end

if istable(Tbl)
    Tbl = table2timetable(Tbl); 
end

data = Tbl{IN_SAMPLE_RANGE,["open","high","low","close","volume","amount"]};

mu = mean(data,1,"omitmissing");
sig = std(data,0,1,"omitmissing");

z = dlarray(reshape(clip((data - mu) ./ (sig+1e-5),-5,5),1,[],6), "BTC");

if(IN_SAMPLE_RANGE(end)+PRED_LEN<=height(Tbl))
    t_vec = Tbl.Time(IN_SAMPLE_RANGE(1):IN_SAMPLE_RANGE(end)+PRED_LEN);
elseif(isnat(future_times))
    t_vec = Tbl.Time(IN_SAMPLE_RANGE);
    temp_mean_diff = mean(diff(t_vec));
    t_vec = [t_vec;t_vec(end) + (temp_mean_diff:temp_mean_diff:PRED_LEN*temp_mean_diff)'];
else
    t_vec = [Tbl.Time(IN_SAMPLE_RANGE);future_times];
end

minutes = minute(t_vec);
hours   = hour(t_vec);
days    = day(t_vec);
months  = month(t_vec);
% Weekday 转换
% MATLAB: 1=Sun, 2=Mon, ..., 7=Sat
% Python: 0=Mon, ..., 6=Sun
wd_matlab = weekday(t_vec); 
weekdays = mod(wd_matlab + 5, 7); 
% [T, B, C]
% [Minute, Hour, Weekday, Day, Month]
ts_stamp = dlarray(cat(3, minutes, hours, weekdays, days, months),"TBC");

rng(Kronos_Options.Seed,"twister")
if(Kronos_Options.Type=="Prediction")
    [predictions_mean, ~, ~] = utility.kronos_inference(model.encoder, model.decoder, model.predictor, model.s2_decoder, ...
        z, ts_stamp, PRED_LEN, Kronos_Options);
    predictions = double(squeeze(predictions_mean).*sig+mu);
    pred_close = predictions(:,4);
    pred_ret = price2ret([data(end,4);pred_close],"Method","continuous");
elseif(Kronos_Options.Type=="Simulation")
    [~, ~, ~, sim_results] = utility.kronos_inference(model.encoder, model.decoder, model.predictor, model.s2_decoder, ...
        z, ts_stamp, PRED_LEN, Kronos_Options);
    predictions = double(squeeze(sim_results).*sig+mu);
    pred_close = mean(predictions(:,4,:),3);
    pred_ret = price2ret([data(end,4);pred_close],"Method","continuous");
end
end