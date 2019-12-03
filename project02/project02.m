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

    % Select the first 15 sample points for each side
    num = 15;
    tau = (0:num-1) / Fs;
    Rt1 = zeros(1, num); % Positive side (for 0~90)
    Rt2 = zeros(1, num); % Negative side (for 90~180)
    yf = fft(y);
    rho = 0.648;

    for iter=1:num
        c_pos = yf(:, 1) .* conj(yf(:, 2));
        c_pos = c_pos ./ (abs(yf(:, 1)) .* abs(yf(:, 2))).^ rho;
        c_neg = yf(:, 2) .* conj(yf(:, 1));
        c_neg = c_neg ./ (abs(yf(:, 1)) .* abs(yf(:, 2))).^ rho;
        ks = 0:length(yf(:, 1))-1;
        es = exp(1i*2*pi*(iter-1).*ks/length(ks));
        c_pos = c_pos .* es.';
        Rt1(iter) = sum(c_pos);
        c_neg = c_neg .* es.';
        Rt2(iter) = sum(c_neg);    
    end

    tau = (-num+1:num-1)/Fs;
    Rt = [Rt2(end:-1:2), Rt1]; % Merge both sides. Totally 29 Samples with center tau=0
    if(TRAINING_SET == 1)
        subplot(4, 4, n)
        plot(tau, abs(Rt))
        xlabel(n)
    end
    [rs, idx] = sort(abs(Rt), 'descend');
    topmany = 3; % Find the top 3 greatest values and indices
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
    err_mean_without_outlier = (sum(e)-max(e))/(length(e)-1) % Avg after deleting greatest error
else
    dlmwrite("result.txt", deg,'delimiter',  '\n', 'precision', '%.7f');
    disp("Output file 'result.txt' created.");
end