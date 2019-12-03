clear;clc

TRAINING_SET = 1;
% For training result, please set TRAINING_SET to 1
% For test output, set TRAINING_SET to 0
% You may change the directory below (PATH_NAME)

sample_num = 0;
PATH_NAME = "";
if(TRAINING_SET == 1)
    sample_num = 14;
    ground_truth = textread("./train/angle.txt");
    PATH_NAME = "./train/";
else
    sample_num = 140;
    PATH_NAME = "./test/";
end
deg_init = ones(1, sample_num);
deg = ones(1, sample_num);

for n = 1:sample_num
    [y, Fs] = audioread(PATH_NAME+mat2str(n)+".wav");
    L = length(y(:, 1));
    % t = (0:L-1)/Fs;

    num = 15;
    tau = (0:num-1) / Fs;
    Rt1 = zeros(1, num);
    Rt2 = zeros(1, num);
    yf = fft(y);
    alpha = 0.648;

    for iter=1:num
        cc1 = yf(:, 1) .* conj(yf(:, 2));
        cc1 = cc1 ./ (abs(yf(:, 1)) .* abs(yf(:, 2))).^ alpha;
        cc2 = yf(:, 2) .* conj(yf(:, 1));
        cc2 = cc2 ./ (abs(yf(:, 1)) .* abs(yf(:, 2))).^ alpha;
        ks = 0:length(yf(:, 1))-1;
        es = exp(1i*2*pi*(iter-1).*ks/length(ks));
        cc1 = cc1 .* es.';
        Rt1(iter) = sum(cc1);
        cc2 = cc2 .* es.';
        Rt2(iter) = sum(cc2);    
    end

    tau = (-num+1:num-1)/Fs;
    Rt = [Rt2(end:-1:2), Rt1];
    if(TRAINING_SET == 1)
        subplot(4, 4, n)
        plot(tau, abs(Rt))
        xlabel(n)
    end
    [rs, idx] = sort(abs(Rt), 'descend');
    topmany = 3;
    idx_top = tau(idx(1:topmany));
    rt_top = Rt(idx(1:topmany));
    avg = dot(idx_top, rt_top) / sum(rt_top);
    deg_init(n) = real(avg * 3430);
    deg(n) = abs(acosd(deg_init(n)));
    if(deg(n) > 180)
        deg(n) = 180;
    end
end


if(TRAINING_SET == 1)
    ground_truth'
    deg
    e = abs(ground_truth' - deg)
    err_mean = mean(e)
    err_mean_without_outlier = (sum(e)-max(e))/(length(e)-1)
else
    dlmwrite("result.txt", deg,'delimiter',  '\n', 'precision', '%.7f');
    disp("Output file 'result.txt' created.");
end