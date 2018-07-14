clear
clc
addpath('functions');
s = RandStream('mt19937ar', 'Seed', 73);
RandStream.setGlobalStream(s);
s.reset;
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
FFTLength = 64;
CPLength = 4;
SymbolsPerFrame = 64;
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
    'CyclicPrefixLength',   CPLength, ...
    'NumSymbols',           SymbolsPerFrame, ...
    'NumTransmitAntennas',  1);

ofdm4QAMDemod = comm.OFDMDemodulator(ofdmQpskMod);

B = 1.4e6;
T = 1/B; % Sample period
Ts = T * (FFTLength + CPLength); %Symbol period.

% Fading parameters
A = -20; % difference between maximum and negligible path power. dB
A_linear = 10^(A/10);
tau_d = 0.75*T; % RMS delay spread
T_m = -tau_d*log(A_linear); % Maximum delay spread. s
f_0 = 1/T_m; % coherence bandwidth. Hz

C = physconst('light'); %ms
f0 = 1e9; %hz
v = 0.0000001; %ms
fd = v/(C/f0); %Hz
T_0 = 9/(16.*pi.*fd); %0.5 coherence time.[s]
% T_0 = 1/fd; % coherence time. s
% T_0 = 0.423/fd;


pathDelays = [0,1,2].*T;
p = (1./tau_d).*exp(-1.*pathDelays./tau_d);
figure;
stem(pathDelays, p);
g = sqrt(T.^2.*p);
% g = [1,0.82,0.67];
pathGains = 10.*log10(g);
rayChan = comm.RayleighChannel( ...
        'PathDelays', pathDelays, ...
        'AveragePathGains', pathGains, ...
        'NormalizePathGains', true, ...
        'PathGainsOutputPort', true, ...
        'MaximumDopplerShift', fd, ...
        'SampleRate', B, ...
        'DopplerSpectrum', doppler('Jakes'), ...
        'RandomStream', 'mt19937ar with seed', ...
        'Seed', 73);
% 
ricChan=comm.RicianChannel( ...
    'PathDelays', pathDelays, ...
    'AveragePathGains', pathGains, ...
    'NormalizePathGains', true, ...
    'PathGainsOutputPort', true, ...
    'MaximumDopplerShift', fd, ...
    'KFactor', 1, ...
    'DirectPathDopplerShift', 0, ...
    'SampleRate', B, ...
    'DopplerSpectrum', doppler('Jakes'), ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', 73);
    
% multipathChan.Visualization = 'Impulse and frequency responses';
% multipathChan.SamplesToDisplay = '100%';

gainScope = dsp.TimeScope( ...
    'SampleRate', rayChan.SampleRate, ...
    'TimeSpan', SymbolsPerFrame/rayChan.SampleRate, ...
    'Name', 'Multipath Gain', ...
    'ShowGrid', true, ... 
    'YLimits', [-40, 10], ...
    'YLabel', 'Gain (dB)');

x1=newRandomBinaryFrame(frameSize);
   
Tx = reshape(QPSKmod(x1), [FFTLength,SymbolsPerFrame]);
Tx = ofdmQpskMod(Tx);

Tx_k = Tx./sqrt(var(Tx)).*sqrt(3*mean(g./max(g)));
TxVar = var(Tx);

% Out = [];
Out = zeros(1, length(Tx));
for i = 1:1
%     [Tx_rayleigh, Taps] = multipathChan(Tx); 
%     Rx = reshape(ofdm4QAMDemod(Tx_rayleigh), [FFTLength*SymbolsPerFrame 1]);
%     scatterplot(Rx, [], [], 'r+');
    [RayleighOut, rayPathGain] = rayChan(Tx);
    % Rician Test?
    In = zeros(1,3);
    s.reset;
    for j = 1:length(Tx)
        if(j >= 3)
            In(1) = Tx(j-2);
        else
            In(1) = 0;
        end
        if (j >= 2)
            In(2) = Tx(j-1);
        else
            In(2) = 0;
        end
        In(3) = Tx(j);
        RAND = randn(1,4);
        % Rayleigh fading
%         Out(j) = rayPathGain(j,3)*In(1) + rayPathGain(j,2)*In(2) + rayPathGain(j,1)*In(3);
        % Rician Fading?
        Out(j) = complex(0.8594,-0.1891)*Tx(j) + rayPathGain(j,3)*In(1) + rayPathGain(j,2)*In(2) + rayPathGain(j,1)*In(3);
    end
    Out = Out.';
%     [~, rayPathGain] = ricChan(Tx);
%     ricChan.reset;
    [RicianOut, ricPathGain] = ricChan(Tx);
end
R = RicianOut-RayleighOut;
% This changes with time where the doppler shift of the spectral components
% influences it.
Z = R./Tx;
% X = sqrt(1.3331).*complex(randn,randn);
PathGains = 10*log10(abs([rayPathGain(:,1), ricPathGain(:,1)].^2));
% PathGains(:,1) = PathGains(:,1)-10*log10(abs(X).^2);
gainScope(PathGains);

RxRay = reshape(ofdm4QAMDemod(RayleighOut), [FFTLength*SymbolsPerFrame 1]);
scatterplot(RxRay);
RxVar = reshape(ofdm4QAMDemod(Out), [FFTLength*SymbolsPerFrame 1]);
scatterplot(RxVar);
RxRic = reshape(ofdm4QAMDemod(RicianOut), [FFTLength*SymbolsPerFrame 1]);
scatterplot(RxRic);
Y = abs(RxRic-RxVar);
% gainScope(10*log10(abs([(rayPathGain(:,1)), ricPathGain(:,1)]).^2));
% figure;
% stem(pathDelays, mean(Taps.*conj(Taps)));
% % stem(pathDelays, mean(out));
% % stem(pathDelays, mean(Taps.*conj(Taps)));
%% MIMO-OFDM_Wireless_Communications_with_MATLAB Playground
% fm = 100; scale = 1e-6; % Maximum Doppler frequency and mu
% ts_mu = 50; ts = ts_mu*scale; fs = 1/ts; % Sampling time/ frequency
% Nd=1e6; % Number of samples
% % obtain the complex fading channel
% [h, Nfft, Nifft, doppler_coeff] = FWGN_model(fm, fs, Nd);
% subplot(211), plot([1:Nd]*ts, 10*log10(abs(h)));
% str=sprintf('Clarke/Gan Model, f_m=%d[Hz], T_s=%d[us]', fm, ts_mu);
% title(str), axis([0 0.5 -30 5])
% subplot(223), hist(abs(h), 50), subplot(224, hist(angle(h), 50));
%% Simulation of Communication Systems Playground
% % Defining some useful variables
% B = 1.4e6;
% Ts = 1/B;
% FFTLength = 64;
% CPLength = 4;
% SymbolsPerFrame = 128;
% BitsPerSymbol = 2;
% frameSize = FFTLength*SymbolsPerFrame*BitsPerSymbol;
% 
% % Fading parameters
% A = -20; % difference between maximum and negligible path power. dB
% A_linear = 10^(A/10);
% tau_d = 0.75*Ts; % RMS delay spread
% T_m = -tau_d*log(A_linear); % Maximum delay spread. s
% f_0 = 1/T_m; % coherence bandwidth. Hz
% 
% C = physconst('light'); %ms
% f0 = 1e9; %hz
% v = 10; %ms
% fd = v/(C/f0); %Hz
% T_0 = 9/(16.*pi.*fd); %0.5 coherence time.[s]
% % T_0 = 1/fd; % coherence time. s
% % T_0 = 0.423/fd;
% 
% pathDelays = [0,1,2].*Ts;
% delta_t = [0:Ts/1000:5.*Ts];
% p = (1./tau_d).*exp(-1.*pathDelays./tau_d);
% ps = (1./tau_d).*exp(-1.*delta_t./tau_d);
% 
% nLength = 3;
% Nd = frameSize/BitsPerSymbol + CPLength*SymbolsPerFrame;
% [h, Nfft, Nifft, doppler_coeff] = mod_FWGN_model(fd, B, Nd, ps, nLength, delta_t, Ts);
% H = h.';
% % H = h.'.*a;
% % figure;
% % plot([1:Nd]*Ts, 10*log10(abs(h*g(1))));
% % [h1, Nfft, Nifft, doppler_coeff] = FWGN_model(fd, B, Nd);
% % % figure;
% % % plot([1:Nd]*Ts, 10*log10(abs(h1*g(2))));
% % [h2, Nfft, Nifft, doppler_coeff] = mod_FWGN_model(fd, B, Nd, ps, nLength, delta_t);
% % figure;
% % plot([1:Nd]*Ts, 10*log10(abs(h2*g(2))));
% % H = [h;h1;h2].'.*g;
% 
% 
% 
% % mLength = nLength;
% % R = zeros(nLength);
% % for n = 1:nLength
% %     for m = 1:mLength
% %         for k = 1:length(ps)
% %            R(n,m) = R(n,m) + ps(k).*sinc(delta_t(k)./Ts - m).*sinc(delta_t(k)./Ts -n).*Ts/1000;
% %         end
% %     end
% % end
% % L = chol(R);
% % Z = (randn(1,nLength) + 1i.*randn(1,nLength))./sqrt(2);
% % g = L * Z.';
% % a = abs(g)./max(abs(g));
% % stem(a)
% % pathGains = 10.*log10(a);
% 
% 
% % g = sqrt(Ts.^2.*p);
% % g = g./max(g);
% % % g = [1,0.82,0.67];
% % pathGains = 10.*log10(g);
% % 
% % 
% % Nd = frameSize/BitsPerSymbol + CPLength*SymbolsPerFrame;
% % [h, Nfft, Nifft, doppler_coeff] = FWGN_model(fd, B, Nd);
% % % figure;
% % % plot([1:Nd]*Ts, 10*log10(abs(h*g(1))));
% % [h1, Nfft, Nifft, doppler_coeff] = FWGN_model(fd, B, Nd);
% % % figure;
% % % plot([1:Nd]*Ts, 10*log10(abs(h1*g(2))));
% % [h2, Nfft, Nifft, doppler_coeff] = FWGN_model(fd, B, Nd);
% % % figure;
% % % plot([1:Nd]*Ts, 10*log10(abs(h2*g(2))));
% % H = [h;h1;h2].'.*g;
% % % 
% % % K = 3
% % % for i=1:size(H,2)
% % %     for j = 1:K
% % %         S(j,i) = sinc(pathDelays(i)/Ts - j);
% % %     end
% % % end
% % % S( abs(S) < 1e-10 ) = 0;
% % % % asdf = H(i,:);
% % % A = [];
% % % for i = 1:size(H,1)
% % %     A = [A; (S * H(i,:).').'];
% % % end
% % % stem(pathDelays, mean(A.*conj(A)));
% % 
% % 
% x1=newRandomBinaryFrame(frameSize);
% 
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
%     'CyclicPrefixLength',   CPLength, ...
%     'NumSymbols',           SymbolsPerFrame, ...
%     'NumTransmitAntennas',  1);
% ofdm4QAMDemod = comm.OFDMDemodulator(ofdmQpskMod);
% 
% Tx = reshape(QPSKmod(x1), [FFTLength,SymbolsPerFrame]);
% Tx = ofdmQpskMod(Tx);
% 
% Tx = [zeros(size(H,2)-1,1); Tx; zeros(size(H,2)-1,1)];
% for i = 1:Nd
%     Tx(i) = H(i,1)*Tx(i) + H(i,2)*Tx(i+1) + H(i,3)*Tx(i+2);
% end
% Tx = Tx(1:Nd);
% 
% 
% Rx = reshape(ofdm4QAMDemod(Tx), [FFTLength*SymbolsPerFrame 1]);
% scatterplot(Rx, [], [], 'r+');
% figure;
% stem(pathDelays, mean(H.*conj(H)));

