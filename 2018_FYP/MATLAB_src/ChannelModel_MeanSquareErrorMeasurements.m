clear all
close all
clc

addpath('functions');

% Defining some useful variables
FFTLength = 2^10;
SymbolsPerFrame = 5;
BitsPerSymbol = 2;

muVec = [0.01, 0.05, 0.1, 0.2, 0.35];
% muVec = [0.001, 0.01, 0.02, 0.03, 0.04];

EbNoVec = (0:13)';
frameSize = FFTLength*SymbolsPerFrame*BitsPerSymbol;
C = 299792458; % universal constant in m/s

figureA = figure(1);
figureB = figure(2);
figureC = figure(3);



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
w = (randn + 1i*randn);
% W = randn(FFTLength, 1) + 1i.*randn(FFTLength,1);
for i = 1:FFTLength
    W(i,1) = w;
end

%% Channel model
maxJE = 0;
maxJ = 0;
maxD = 0;
for k = 1:length(muVec)
    cumulative_errors = 0;
    cumulative_errors_noise_only = 0;
    trained = 0;
    E = [];
    J_vec = [];
    D_vec = [];
    JE_vec = [];

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

    mu = muVec(k);
    % the number of iterations for the slowest converging
    % series to converge to within 1*10-10 of the mean
    n_max = -3/log10(1-mu*R);       
    for i = 1:max(floor(n_max/SymbolsPerFrame),1)
        for j = 1:SymbolsPerFrame
            e(:,j) = knownOFDM(:,j) - conj(H_hat).*equalisedOFDM(:,j);
%             H_hat = H_hat + (0.5./(0.5+abs(equalisedOFDM(:,j)).^2)).*equalisedOFDM(:,j).*conj(e(:,j));
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
        end
    end
    figure(figureA);
    hold on
    plot(J_vec);
    title("Mean square error vs training symbols");
    ylabel("Mean square error");
    xlabel("training symbols");
    if(maxJ < max(J_vec)); maxJ = max(J_vec); else; maxJ=maxJ; end
    axis([0,100,0,maxJ]);
    figure(figureB);
    hold on
    plot(D_vec);
    title("Mean square deviation vs training symbols");
    ylabel("Mean square deviation");
    xlabel("training symbols");
    if(maxD < max(D_vec)); maxD = max(D_vec); else; maxD=maxD; end
    axis([0,100,0,maxD]);
    figure(figureC);
    plot(JE_vec);
    hold on
    title("Excess mean square error vs training symbols");
    ylabel("Excess mean square error");
    xlabel("training symbols");   
    if(maxJE < max(JE_vec)); maxJE = max(JE_vec); else; maxJE=maxJE; end
    axis([0,100,0,maxJE]);
end
string1 = "mu = " + muVec(1);
string2 = "mu = " + muVec(2);
string3 = "mu = " + muVec(3);
string4 = "mu = " + muVec(4);
string5 = "mu = " + muVec(5);
figure(figureA);
legend(string1, string2, string3, string4, string5);
figure(figureB);
legend(string1, string2, string3, string4, string5);
figure(figureC);
legend(string1, string2, string3, string4, string5);
