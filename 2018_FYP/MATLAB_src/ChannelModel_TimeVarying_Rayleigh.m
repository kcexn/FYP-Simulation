clear all;
close all;
clc;
addpath('functions');
%% Defining some useful variables
%Constants
C = physconst('light'); %ms

% Modem variables
M = 4; % M-QAM;
FFTLength = 2^6;
BitsPerSymbol = log2(M);
SymbolsPerFrame = round(1500*8/BitsPerSymbol/FFTLength); % Ethernet v2 MTU?
CPLength = 4;
frameSize = FFTLength*SymbolsPerFrame*BitsPerSymbol;
NoOfFrames = 500;

% Tranmission and Reception Variables

f0 = 1e9; % Carrier frequency [Hz]
v = 0.013; % UE velocity [m/s] (speed of a garden snail?)
B = 1.4e6; % OFDM Symbol Bandwidth [Hz]
T = 1/B; % Sample period [s]
Ts = T * (FFTLength + CPLength); % OFDM Symbol Period [s]

% Equalisation variables
mu = 0.5;
% mu = 0.1;
H_hat = zeros(FFTLength,1);
decision_directed = false;
training_symbols = 3;

% Wireless Channel Variables
% EbNoVec = (10:1:30)';
EbNoVec = [10,11,12,13,14,15,16];
EbNo = 22;
snr = EbNo + 10*log10(BitsPerSymbol);

% Fading parameters
A = -20; % difference between maximum and negligible path power. dB
A_linear = 10^(A/10);
tau_d = 0.75*T; % RMS delay spread
T_m = -tau_d*log(A_linear); % Maximum delay spread. s
f_0 = 1/T_m; % coherence bandwidth. Hz

fd = v/(C/f0); % Doppler frequency [Hz]
T_0 = 9/(16.*pi.*fd); % 0.5 coherence time.[s]

pathDelays = [0,1,2].*T;
p = (1./tau_d).*exp(-1.*pathDelays./tau_d);
g = sqrt(T.^2.*p);
pathGains = 10.*log10(g);

%% Evaluation variables
berVec = zeros(1, length(EbNoVec));
berTmpVec = zeros(1,length(NoOfFrames));
%% Defining System Objects
% QPSK modulation
QPSKmod = comm.QPSKModulator('BitInput', true);
qpskDemod = comm.QPSKDemodulator('BitOutput',true);

% OFDM Modulation
ofdmQpskMod = comm.OFDMModulator( ...
    'FFTLength',            FFTLength, ...
    'NumGuardBandCarriers', [0;0], ...
    'InsertDCNull',         false, ...
    'PilotInputPort',       false, ...
    'CyclicPrefixLength',   CPLength, ...
    'NumSymbols',           SymbolsPerFrame, ...
    'NumTransmitAntennas',  1);
ofdm4QAMDemod = comm.OFDMDemodulator(ofdmQpskMod);

% rayChan = comm.RayleighChannel( ...
%         'PathDelays', pathDelays, ...
%         'AveragePathGains', pathGains, ...
%         'NormalizePathGains', true, ...
%         'PathGainsOutputPort', true, ...
%         'MaximumDopplerShift', fd, ...
%         'SampleRate', B, ...
%         'DopplerSpectrum', doppler('Jakes'));
    
% AWGN Channel
awgnChannel = comm.AWGNChannel( ...
    'NoiseMethod', 'Variance', ...
    'VarianceSource', 'Input port');
    
ricChan=comm.RicianChannel( ...
    'PathDelays', pathDelays, ...
    'AveragePathGains', pathGains, ...
    'NormalizePathGains', true, ...
    'PathGainsOutputPort', true, ...
    'MaximumDopplerShift', fd, ...
    'KFactor', 3, ...
    'DirectPathDopplerShift', 0, ...
    'DirectPathInitialPhase', 0, ...
    'SampleRate', B, ...
    'DopplerSpectrum', doppler('Jakes'));    

% multipathChan = rayChan;
multipathChan = ricChan;

% gainScope = dsp.TimeScope( ...
%     'SampleRate', multipathChan.SampleRate, ...
%     'TimeSpan', SymbolsPerFrame/multipathChan.SampleRate, ...
%     'Name', 'Multipath Gain', ...
%     'ShowGrid', true, ... 
%     'YLimits', [-40, 10], ...
%     'YLabel', 'Gain (dB)');

%% Simulation
for k = 1:length(EbNoVec)
    % Simulation
    for i = 1:NoOfFrames
        decision_directed = false;
        x1=newRandomBinaryFrame(frameSize);
        Tx_QPSK = reshape(QPSKmod(x1), [FFTLength,SymbolsPerFrame]);
        Tx = ofdmQpskMod(Tx_QPSK);

        % AWGN channel system variables       
        snr = EbNoVec(k) + 10*log10(BitsPerSymbol); % Only when sweeping through SNR's
        powerDB = 10*log10(var(Tx));
        noiseVar = 10.^(0.1*(powerDB-snr));

        [TxMultipath, multipathTaps] = multipathChan(Tx);
        TxMultipath = awgnChannel(TxMultipath, noiseVar);
        Rx = ofdm4QAMDemod(TxMultipath);
%         figure();
%         scatterplot(reshape(Rx, [FFTLength*SymbolsPerFrame,1]));
        % Equaliser
        e = zeros(FFTLength, size(Rx, 2)); % Initialising e (not necessary but a small optimisation)
        H_hat = conj(mean(1./(Rx(:,1:3)./Tx_QPSK(:,1:3)),2));
%         H_hat = ((conj(H_hat)*H_hat.').' + noiseVar./B.*eye(size((conj(H_hat)*H_hat.').')))^(-1)*conj(H_hat);
        for j = 1:size(Rx,2)
            if(~decision_directed)
                e(1:FFTLength,j) = Tx_QPSK(:,j) - conj(H_hat).*Rx(:,j);
                H_hat = H_hat + (mu./(0.005+abs(Rx(:,j)).^2)).*Rx(:,j).*conj(e(:,j));
%                 H_hat = H_hat + mu.*Rx(:,j).*conj(e(:,j));
                Rx(:,j) = conj(H_hat).*Rx(:,j);
                if(j == training_symbols)
                    decision_directed = ~decision_directed;
                end
            else
                equalisedRx = conj(H_hat).*Rx(:,j);
    %             scatterplot(Rx(:,j));
                dd = qpskDemod(equalisedRx);
                dd = QPSKmod(dd);
%                 figure();
%                 scatterplot(dd);
                e(1:FFTLength, j) = dd - equalisedRx;
                H_hat = H_hat + (mu./(0.005+abs(Rx(:,j)).^2)).*Rx(:,j).*conj(e(:,j));
%                 H_hat = H_hat + mu.*Rx(:,j).*conj(e(:,j));
                Rx(:,j) = equalisedRx;
            end
        end
%         figure();
%         scatterplot(reshape(Rx, [FFTLength*SymbolsPerFrame 1]));
        Rx = qpskDemod(reshape(Rx, [FFTLength*SymbolsPerFrame 1]));
        berTmpVec(i) = sum(Rx ~= x1)/frameSize; % Average this to get a sense of BER
    end
    berVec(k) = mean(berTmpVec);
end
figure();
semilogy(EbNoVec, berVec);
title("BER vs EbNo");
xlabel("E_b/N_0 [dB]");
ylabel("BER");





