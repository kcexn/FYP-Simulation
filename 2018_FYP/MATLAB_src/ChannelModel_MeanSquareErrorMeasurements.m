clear all
close all
clc

addpath('functions');

% Defining some useful variables
FFTLength = 2^8;
SymbolsPerFrame = 5;
BitsPerSymbol = 2;

EbNoVec = (0:13)';
frameSize = FFTLength*SymbolsPerFrame*BitsPerSymbol;
C = 299792458; % universal constant in m/s



%% Defining all the channel model components.
% QPSK modulation
QPSKmod = comm.QPSKModulator('BitInput', true);
qpskDemod = comm.QPSKDemodulator('BitOutput',true);

% OFDM Modulation
ofdmQpskMod = comm.OFDMModulator( ...
    'FFTLength',            FFTLength, ...
    'NumGuardBandCarriers', [0;0], ...
    'InsertDCNull',         false, ...
    'PilotInputPort',       false, ...
    'CyclicPrefixLength',   0, ...
    'NumSymbols',           SymbolsPerFrame, ...
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

    % Generate 64 realizations of a sing,le tap gain for
    % ensemble averaging
w = randn + 1i*randn;
for i = 1:FFTLength
    W(i,1) = w;
end

%% Channel model
cumulative_errors = 0;
cumulative_errors_noise_only = 0;
trained = 0;
E = [];
J_vec = [];
D_vec = [];
JE_vec = [];

for k = 1:1
    x1 = newRandomBinaryFrame(frameSize);

    qpskTx = reshape(QPSKmod(x1), [FFTLength,SymbolsPerFrame]);
    %AWGN only
    qpskOFDMTx = ofdmQpskMod(qpskTx);
    % pre-alter the symbols with a channel response.
    for i = 1:SymbolsPerFrame
       qpskTx(:,i) = W.*qpskTx(:,i); 
    end
    % AWGN and channel filter
    qpskOFDM_H_Tx = ofdmQpskMod(qpskTx);

    powerDB_AWGNOnly = 10*log10(var(qpskOFDMTx));
    noiseVar_AWGNOnly = 10.^(0.1*(powerDB_AWGNOnly-snr));

    qpsk_OFDM_Noise_Tx = awgnChannel(qpskOFDMTx, noiseVar_AWGNOnly);
    qpsk_OFDM_H_Noise_Tx = awgnChannel(qpskOFDM_H_Tx, noiseVar_AWGNOnly);

    %% Equaliser

    equalisedOFDM = ofdm4QAMDemod(qpsk_OFDM_H_Noise_Tx);
    knownOFDM = ofdm4QAMDemod(qpskOFDMTx);

    % LMS estimator.
    
    H_hat = zeros(FFTLength,1);
    R = var(equalisedOFDM(1,:));
    S = 2/R;
    %         Expectation of W(n) = W_o - W_o(1-mu*var)^n
    %         Expectation of W(n) = 1/W - 1/W(1-mu*var)^n        
    mu = 0.05;
    % the number of iterations for the slowest converging
    % series to converge to within 1*10-10 of the mean
    n_max = -3/log10(1-mu*R);       
    for i = 1:max(floor(n_max/SymbolsPerFrame),1)
        for j = 1:SymbolsPerFrame
            e(:,j) = knownOFDM(:,j) - conj(H_hat).*equalisedOFDM(:,j);
            H_hat = H_hat + mu.*equalisedOFDM(:,j).*conj(e(:,j));
            
            error(:,j) = knownOFDM(:,j) - equalisedOFDM(:,j)./W;
            Wiener_minimum_mean_square_error = var(error(:,j));
            % mean square error
            J = var(e(:,j));
            J_Excess = J - Wiener_minimum_mean_square_error;
            JE_vec = [JE_vec, J_Excess];
            
            minimum_mean_square_weight_error = mu.*Wiener_minimum_mean_square_error./(2-mu*R) + (1-mu*R).^(2*n_max).*(abs(1./W(1)).^2 - (mu*Wiener_minimum_mean_square_error./(2-mu*R)));
            % mean square deviation
            D = mean(abs(1./W - conj(H_hat)).^2);
%             D = var((1./W - conj(H_hat)));
            
            J_vec = [J_vec, J];
            D_vec = [D_vec, D];
        %         D = var(1./W - conj(H_hat));
%             D
%             J_Excess
%             for i = 1:31
%                 equalisedOFDMtmp(1:64,i) = conj(H_hat).*equalisedOFDM(1:64,i);
%             end
%             equalisedOFDMtmp = ofdmQpskMod(equalisedOFDMtmp); 
%             qpskRxtmp = reshape(ofdm4QAMDemod(equalisedOFDMtmp), [1984 1]);
%             scatterplot(qpskRxtmp);
        end
    end


    %% back to channel model
%     qpskRx2 = reshape(ofdm4QAMDemod(equalisedOFDM), [1984 1]);
% %     qpskRx1 = reshape(ofdm4QAMDemod(qpsk_OFDM_Noise_Tx), [1984 1]);
%     qpskRx1 = reshape(ofdm4QAMDemod(qpsk_OFDM_H_Noise_Tx), [1984 1]);
%     s = scatterplot(qpskRx1);
%     hold on
%     scatterplot(qpskRx2, [], [], 'rx', s);
%     dataOut = qpskDemod(qpskRx1);
%     dataOut2 = qpskDemod(qpskRx2);
%     hold off
% 
%     cumulative_errors_noise_only = cumulative_errors_noise_only + length(dataOut(dataOut ~= x1));
%     cumulative_errors = cumulative_errors + length(dataOut2(dataOut2 ~= x1));
% %     E = [E, excess];
end

% average_E = mean(E);
% cumulative_bit_error_rate = cumulative_errors/(k.*frameSize);
% cumulative_noise_only_error_rate = cumulative_errors_noise_only/(k.*frameSize);
