clear
clc
addpath('functions');
%% Test script for learning about MATLAB functions
% % pnSequence
%     pnSequenceQPSK = comm.PNSequence( ...
%         'Polynomial', [12 6 4 0], ...
%         'SamplesPerFrame', 3968, ...
%         'InitialConditions', [0 1 0 0 1 0 1 0 0 0 1 1]);
%     x1 = pnSequenceQPSK();
%     % QPSK modulation
%     QPSKmod = comm.QPSKModulator( ...
%         'BitInput', true ...        
%         );
%     qpskTx = QPSKmod(x1);
%     qpskPlotTx = qpskTx;
%     
%     qpskTx = reshape(qpskTx, [64 31]);
%     
%     % OFDM Modulation
%     ofdmQpskMod = comm.OFDMModulator( ...
%         'FFTLength',            64, ...
%         'NumGuardBandCarriers', [0;0], ...
%         'InsertDCNull',         false, ...
%         'PilotInputPort',       false, ...
%         'CyclicPrefixLength',   0, ...
%         'NumSymbols',           31, ...
%         'NumTransmitAntennas',  1);
% 
%     qpskTx = ofdmQpskMod(qpskTx);
%     
%     % AWGN Channel
%     % SNR in dB
%     EbNo = 8;
%     % SNR taking into account 2 bits per QAM symbol?
% 
%     % The addition of the 10*log10(2) comes straight out of the
%     % definition of SNR in relation to EbN0
%     % Eb/N0 = (S/N)(W/Rb)
%     snr = EbNo + 10*log10(2);
% 
%     powerDB = 10*log10(var(qpskTx));
%     noiseVar = 10.^(0.1*(powerDB-snr));
%     
%     awgnChannel = comm.AWGNChannel( ...
%         'NoiseMethod', 'Variance', ...
%         'VarianceSource', 'Input port' ...
%         );
%     qpskTx = awgnChannel(qpskTx, noiseVar);
%     
%     % OFDM Demodulator
%     ofdm4QAMDemod = comm.OFDMDemodulator(ofdmQpskMod);
%     qpskRx = ofdm4QAMDemod(qpskTx);
%     
%     qpskRx = reshape(qpskRx, [1984, 1]);
% %     H = scatterplot(qpskRx, [], [], 'b.');
% %     hold on
% %     scatterplot(qpskPlotTx, [], [], 'r*', H);
%     
%     %QPSK Demodulator
%     qpskDemod = comm.QPSKDemodulator( ...
%         'BitOutput',true ...
%         );
%     dataOut = qpskDemod(qpskRx);
% % figure,
% NFFT = 64;
% t=[-50:49];
% y = sin(2*pi*.1*t) ;
% plot(t,y);
% Y = fftshift(fft(y, NFFT));
% nVals = (-NFFT/2:NFFT/2-1)/NFFT;
% plot(nVals, abs(Y));
% plot([-50:49]/(2*pi),abs(Y));

%% FFT Plotting and stuff

% NFFT = 1000;
% Ts = 0.001;
% fs = 1/Ts;
% t = [-0.02:Ts:0.02-Ts];
% x = cos(2*pi*50*t);
% figure();
% plot(t,x);
% X = fftshift(fft(x,NFFT));
% fVals = fs*(-NFFT/2:NFFT/2-1)/NFFT;
% figure();
% subplot(2,1,1);
% plot(fVals, abs(X)/(0.02/Ts));
% subplot(2,1,2);
% plot(fVals, angle(X));
% 
% NFFT = 1000;
% Ts = 0.001;
% fs = 1/Ts;
% t = [-0.02+0.0971:Ts:0.02-Ts+0.0971];
% x = cos(2*pi*50*t);
% figure();
% plot(t,x);
% X = fftshift(fft(x,NFFT));
% fVals = fs*(-NFFT/2:NFFT/2-1)/NFFT;
% figure();
% subplot(2,1,1);
% plot(fVals, abs(X)/(0.02/Ts));
% subplot(2,1,2);
% plot(fVals, angle(X));

%% Looking at Channel Tap Weight Time domain response.

% % Defining some useful variables
% EbNoVec = (0:13)';
% berTheoryVecQPSK= [];
% berVecQPSK = [];
% frameSize = 3968;
% alpha = (pi/5).*1e3.*0.6; % a scaling factor to adjust for antenna directional pattern
% r = (pi/5).*1e3; % 1km in m
% f = 900e3; % 0.9GHz in Hz
% C = 299792458; % universal constant in m/s
% 
% x1 = newRandomBinaryFrame(frameSize);
% 
% % QPSK modulation
% QPSKmod = comm.QPSKModulator('BitInput', true);
% qpskDemod = comm.QPSKDemodulator('BitOutput',true);
% % OFDM Modulation
% ofdmQpskMod = comm.OFDMModulator( ...
%     'FFTLength',            64, ...
%     'NumGuardBandCarriers', [0;0], ...
%     'InsertDCNull',         false, ...
%     'PilotInputPort',       false, ...
%     'CyclicPrefixLength',   48, ...
%     'NumSymbols',           31, ...
%     'NumTransmitAntennas',  1);
% ofdm4QAMDemod = comm.OFDMDemodulator(ofdmQpskMod);
% % noise
% awgnChannel = comm.AWGNChannel( ...
%     'NoiseMethod', 'Variance', ...
%     'VarianceSource', 'Input port');
% % Channel Properties
% w = [randn + 1i*randn, randn + 1i*randn, randn + 1i*randn];
% w1 = randn + 1i*randn;
% for i = 1:64
%    w2(i) = randn + 1i*randn;
% end
% fft(w2);
% %
% 
% qpskTx = reshape(QPSKmod(x1), [64,31]);
% qpskOFDMTx = ofdmQpskMod(qpskTx);
% 
% qpskOFDMTx_H = conv(qpskOFDMTx, w2);
% qpskOFDMTx_H = qpskOFDMTx_H(1:(64+48)*31);
% 
% % Add noise
% snr = 11 + 10*log10(2);
% powerDB = 10*log10(var(qpskOFDMTx));
% noiseVar = 10.^(0.1*(powerDB-snr));
% 
% qpsk_OFDM_Noise_Tx = awgnChannel(qpskOFDMTx_H, noiseVar);
% qpskOFDMTx_Noise_Test = awgnChannel(qpskOFDMTx, noiseVar);
% %
% 
% % equalise
% R = ofdm4QAMDemod(qpsk_OFDM_Noise_Tx);
% S = ofdm4QAMDemod(qpskOFDMTx);
% 
% % Zero - forcing estimate.
% H_hat = R(:,1)./S(:,1);
% for i = 1:31
%     Y_hat(1:64,i) = R(1:64,i)./H_hat;
% end
% 
% equalisedOFDM = ofdmQpskMod(Y_hat);
% % equalisedOFDM = reshape(ifft(ifftshift(Y_hat,1)),[1984,1]);
% %
% 
% qpskRx1 = reshape(ofdm4QAMDemod(qpskOFDMTx_Noise_Test), [1984 1]);
% s = scatterplot(qpskRx1);
% hold on;
% 
% qpskRx1 = reshape(ofdm4QAMDemod(equalisedOFDM), [1984 1]);
% s = scatterplot(qpskRx1, [], [], 'rx', s);
% 
% qpskRx1 = reshape(ofdm4QAMDemod(qpsk_OFDM_Noise_Tx), [1984 1]);
% scatterplot(qpskRx1, [], [], 'go', s);
% axis([-20,20,-20,20]);
% 
% dataOut = qpskDemod(qpskRx1);
% 
% isequal(dataOut, x1)

%% Rayleigh fading model
% Defining some useful variables
FFTLength = 2^8;
SymbolsPerFrame = 31;
BitsPerSymbol = 2;
frameSize = FFTLength*SymbolsPerFrame*BitsPerSymbol;

% QPSK modulation
QPSKmod = comm.QPSKModulator('BitInput', true);
qpskDemod = comm.QPSKDemodulator('BitOutput',true);

% OFDM Modulation
ofdmQpskMod = comm.OFDMModulator( ...
    'FFTLength',            FFTLength, ...
    'NumGuardBandCarriers', [0;0], ...
    'InsertDCNull',         false, ...
    'PilotInputPort',       false, ...
    'CyclicPrefixLength',   4, ...
    'NumSymbols',           SymbolsPerFrame, ...
    'NumTransmitAntennas',  1);

ofdm4QAMDemod = comm.OFDMDemodulator(ofdmQpskMod);

B = 1.4e6;
T = 1/B;
C = physconst('light'); %ms
f0 = 1e9; %hz
v = 0.1; %ms
fd = v/(C/f0); %Hz

pathDelays = [0,1,2].*T;
p = (1./T).*exp(-0.7.*pathDelays./T);
g = sqrt(T.^2.*p);
% g = [1,0.82,0.67];
pathGains = 10.*log10(g);
% multipathChan = comm.RayleighChannel( ...
%         'PathDelays', pathDelays, ...
%         'AveragePathGains', pathGains, ...
%         'NormalizePathGains', true, ...
%         'PathGainsOutputPort', true, ...
%         'MaximumDopplerShift', fd, ...
%         'SampleRate', B, ...
%         'DopplerSpectrum', {doppler('Jakes'), doppler('Jakes'), doppler('Jakes')});

multipathChan=comm.RicianChannel( ...
    'PathDelays', pathDelays, ...
    'AveragePathGains', pathGains, ...
    'NormalizePathGains', true, ...
    'PathGainsOutputPort', true, ...
    'MaximumDopplerShift', fd, ...
    'KFactor', 3, ...
    'DirectPathDopplerShift', 0, ...
    'SampleRate', B, ...
    'DopplerSpectrum', doppler('Jakes'));
    
% rayleighChan.Visualization = 'Impulse and frequency responses';
% rayleighChan.SamplesToDisplay = '100%';
    
x1=newRandomBinaryFrame(frameSize);
   
Tx = reshape(QPSKmod(x1), [FFTLength,SymbolsPerFrame]);
Tx = ofdmQpskMod(Tx);
for i = 1:25
    [Tx_rayleigh, Taps] = multipathChan(Tx); 
    Rx = reshape(ofdm4QAMDemod(Tx_rayleigh), [FFTLength*SymbolsPerFrame 1]);
%     figure();
%     scatter(real(Rx), imag(Rx),'+', 'MarkerEdgeColor', [1, 209/255, 220/255]);
%     axis([-2.1 2.1 -2.1 2.1]);
%     xlabel("In Phase");
%     ylabel("Quadrature");
%     title("Scatterplot");
    scatterplot(Rx, [], [], 'r+');
end
%% Modeling of a simple powerdelay profile and its tap gains.
% let's assume a simple exponential diffuse channel model from Jeruchim.
% p(\tau) = \frac{1}{T}e^{-0.4\tau/T}
% T = 1/B where B is the bandwidth of our OFDM symbol
% but lets let T = 1 for now.
% \tau is between 0 and 4 again from the textbook but really we should
% limit this based on where the magnitude of the delay power profile
% becomes below a given threshold.

% %Useful variables
% FFTLength = 2^10;
% SymbolsPerFrame = 31;
% BitsPerSymbol = 2;
% frameSize = FFTLength*SymbolsPerFrame*BitsPerSymbol;
% % QPSK modulation
% QPSKmod = comm.QPSKModulator('BitInput', true);
% qpskDemod = comm.QPSKDemodulator('BitOutput',true);
% 
% % OFDM Modulation
% ofdmQpskMod = comm.OFDMModulator( ...
%     'FFTLength',            FFTLength, ...
%     'NumGuardBandCarriers', [0;0], ...
%     'InsertDCNull',         false, ...
%     'PilotInputPort',       false, ...
%     'CyclicPrefixLength',   5, ...
%     'NumSymbols',           SymbolsPerFrame, ...
%     'NumTransmitAntennas',  1);
% 
% ofdm4QAMDemod = comm.OFDMDemodulator(ofdmQpskMod);
% 
% % channel model
% T = 1;
% tau = [0,1,2,3].*T; %three path delay in microseconds.
% p = (1./T).*exp(-0.4.*tau./T);
% % multiplying a value pulled from a normal distribution scales
% % it's standard deviation.
% x1=newRandomBinaryFrame(frameSize);
% % plot(tau, tmp);
% for i = 1:10
%     g = sqrt(T.^2.*p).*(randn(1,length(tau)) + 1i* randn(1,length(tau)));
%     Tx = reshape(QPSKmod(x1), [FFTLength,SymbolsPerFrame]);
%     Tx = ofdmQpskMod(Tx);
%     Tx = conv(g, Tx);
%     Rx = reshape(ofdm4QAMDemod(Tx(1:end-length(tau)+1)), [FFTLength*SymbolsPerFrame 1]);
%     scatterplot(Rx, [], [], 'r+');
% end

