clear all
close all
clc

addpath('functions');

% Defining some useful variables
EbNoVec = (0:13)';
berTheoryVecQPSK= [];
berVecQPSK = [];
frameSize = 3968;
C = 299792458; % universal constant in m/s

% Testing is a flag for turning on ensemble averaging
Testing = 0;
H = randn(1,1) + randn(1,1)*1i;

%% Defining all the channel model components.
% QPSK modulation
QPSKmod = comm.QPSKModulator('BitInput', true);
qpskDemod = comm.QPSKDemodulator('BitOutput',true);

% OFDM Modulation
ofdmQpskMod = comm.OFDMModulator( ...
    'FFTLength',            64, ...
    'NumGuardBandCarriers', [0;0], ...
    'InsertDCNull',         false, ...
    'PilotInputPort',       false, ...
    'CyclicPrefixLength',   0, ...
    'NumSymbols',           31, ...
    'NumTransmitAntennas',  1);

ofdm4QAMDemod = comm.OFDMDemodulator(ofdmQpskMod);

% AWGN Channel
awgnChannel = comm.AWGNChannel( ...
    'NoiseMethod', 'Variance', ...
    'VarianceSource', 'Input port');

%With noise added
snr = EbNoVec(11) + 10*log10(2);

%% Generate channel response
% W = comm.RayleighChannel( ...
%         'PathDelays', [0.001], ...
%         'AveragePathGains', [0]);

% Generate the frequency domain gains.
if (Testing == 0)
    for i = 1:64
        W(i,1) = randn + 1i*randn;
    end    
else
    % Generate 64 realizations of a single tap gain for
    % ensemble averaging
    w = randn + 1i*randn;
    for i = 1:64
        W(i,1) = w;
    end
end

%% Channel model
cumulative_errors = 0;
cumulative_errors_noise_only = 0;
trained = 0;
E = [];

for k = 1:100
    x1 = newRandomBinaryFrame(frameSize);

    qpskTx = reshape(QPSKmod(x1), [64,31]);
    %AWGN only
    qpskOFDMTx = ofdmQpskMod(qpskTx);
    % pre-alter the symbols with a channel response.
    for i = 1:31
       qpskTx(:,i) = W.*qpskTx(:,i); 
    end
    % AWGN and channel filter
    qpskOFDM_H_Tx = ofdmQpskMod(qpskTx);

    powerDB_AWGNOnly = 10*log10(var(qpskOFDMTx));
    noiseVar_AWGNOnly = 10.^(0.1*(powerDB_AWGNOnly-snr));
    powerDB_H = 10*log10(var(qpskOFDM_H_Tx));
    noiseVar_H = 10.^(0.1*(powerDB_H-snr));

    qpsk_OFDM_Noise_Tx = awgnChannel(qpskOFDMTx, noiseVar_AWGNOnly);
%     qpsk_OFDM_H_Noise_Tx = qpskOFDM_H_Tx;
    qpsk_OFDM_H_Noise_Tx = awgnChannel(qpskOFDM_H_Tx, noiseVar_AWGNOnly);

    %% Equaliser

    equalisedOFDM = ofdm4QAMDemod(qpsk_OFDM_H_Noise_Tx);
    knownOFDM = ofdm4QAMDemod(qpskOFDMTx);

    % Zero - forcing estimate.
%     H_hat_Z = equalisedOFDM(:,1)./knownOFDM(:,1);
%     for i = 1:31
%         e(1:64,i) = knownOFDM(:,i) - (1./W).*equalisedOFDM(:,i);
%     end

    % LMS estimator.
    if ( trained == 0 )    
        trained = 1;        
        H_hat = zeros(64,1);
        R = max(diag(cov(equalisedOFDM.')));
        R_prime = min(diag(cov(equalisedOFDM.')));
        chi = R - R_prime;
        [lambdas, indices] = sort(diag(cov(equalisedOFDM.')));
        convergence_quality = length(lambdas(lambdas > 1))/64;
        S = 2/R;
%         Expectation of W(n) = W_o - W_o(1-mu*var)^n
%         Expectation of W(n) = 1/W - 1/W(1-mu*var)^n
        % mu 0.1
        mu = 0.1;
        % the number of iterations for the slowest converging
        % series to converge to within 1*10-10 of the mean
        n_max = -3/log10(1-mu*R_prime);
%         n_max = 31;
%         for i = 1
        for i = 1:floor(n_max/31)
            for j = 1:31
                e(1:64,j) = knownOFDM(:,j) - conj(H_hat).*equalisedOFDM(:,j);
                H_hat = H_hat + mu.*equalisedOFDM(1:64,j).*conj(e(1:64,j));
            end
        end
        if (Testing == 1)
%         for m = 1
            for m = 1:floor(n_max/31)
                for t = 1:31
                    error(1:64,t) = knownOFDM(:,t) - (1./W).*equalisedOFDM(:,t);
                end
            end
            Wiener_minimum_mean_square_error = mean(abs(error(:,t)).^2);
            % mean square error
            J = mean(abs(e(:,j)).^2);
            J_Excess = J - Wiener_minimum_mean_square_error;
            minimum_mean_square_weight_error = mu.*Wiener_minimum_mean_square_error./(2-mu*R) + (1-mu*R).^(2*n_max).*(abs(1./W(indices(1))).^2 - (mu*Wiener_minimum_mean_square_error./(2-mu*R)));

            % mean square deviation
            D = mean(abs(1./W - conj(H_hat)).^2);
    %         D = var(1./W - conj(H_hat));
            D
            J_Excess
        end
    end
    for i = 1:31
        equalisedOFDM(1:64,i) = conj(H_hat).*equalisedOFDM(1:64,i);
    end
    equalisedOFDM = ofdmQpskMod(equalisedOFDM);
    %% back to channel model

    qpskRx2 = reshape(ofdm4QAMDemod(equalisedOFDM), [1984 1]);
%     qpskRx1 = reshape(ofdm4QAMDemod(qpsk_OFDM_Noise_Tx), [1984 1]);
    qpskRx1 = reshape(ofdm4QAMDemod(qpsk_OFDM_H_Noise_Tx), [1984 1]);
    s = scatterplot(qpskRx1);
    hold on
    scatterplot(qpskRx2, [], [], 'rx', s);
    dataOut = qpskDemod(qpskRx1);
    dataOut2 = qpskDemod(qpskRx2);
    hold off

    cumulative_errors_noise_only = cumulative_errors_noise_only + length(dataOut(dataOut ~= x1));
    cumulative_errors = cumulative_errors + length(dataOut2(dataOut2 ~= x1));
%     E = [E, excess];
end

average_E = mean(E);
cumulative_bit_error_rate = cumulative_errors/(k.*frameSize);
cumulative_noise_only_error_rate = cumulative_errors_noise_only/(k.*frameSize);

% isequal(dataOut, x1)
% isequal(dataOut2, x1)


    