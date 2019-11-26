clear;clc

ground_truth = textread("./train/angle.txt");
deg = ones(1, 14);

for n = 1:14
    [y, Fs] = audioread("./train/"+mat2str(n)+".wav");
    L = length(y(:, 1));
    % t = (0:L-1)/Fs;

    num = 15;
    tau = (0:num-1) / Fs;
    Rt1 = zeros(1, num);
    Rt2 = zeros(1, num);
    yf = fft(y);

    for iter=1:num
        cc1 = yf(:, 1) .* conj(yf(:, 2));
        cc1 = cc1 ./ (abs(yf(:, 1)) .* abs(yf(:, 2)));
        cc2 = yf(:, 2) .* conj(yf(:, 1));
        cc2 = cc2 ./ (abs(yf(:, 1)) .* abs(yf(:, 2)));
        ks = 0:length(yf(:, 1))-1;
        es = exp(1i*2*pi*(iter-1).*ks/length(ks));
        cc1 = cc1 .* es.';
        Rt1(iter) = sum(cc1);
        cc2 = cc2 .* es.';
        Rt2(iter) = sum(cc2);    
    end

    tau = (-num+1:num-1)/Fs;
    Rt = [Rt2(end:-1:2), Rt1];
    subplot(4, 4, n)
    plot(tau, abs(Rt))
    [rs, idx] = sort(abs(Rt), 'descend');
    topmany = 3;
    idx_top = tau(idx(1:topmany));
    rt_top = Rt(idx(1:topmany));
    avg = dot(idx_top, rt_top) / sum(rt_top);
    deg(n) = abs(acosd(avg * 3430));
    if(deg(n) > 180)
        deg(n) = 180;
    end
    xlabel(n)
end

ground_truth'
deg
e = abs(deg - ground_truth')
mean(e)
(sum(e)-max(e))/(length(e)-1)