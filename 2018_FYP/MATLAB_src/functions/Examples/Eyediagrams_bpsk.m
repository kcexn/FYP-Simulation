% Lab 3 ECE 3141 Matlab code to generate eye digrams using Rectangular
% pulses to carry digital information
% Copyright: Gayathri Kongara, ECSE department, Monash University
% Created on 16/01/2017
% Revisions:
clear all
%close all
rng default
SNR = [25];
%Modulation parameters
M =  2;                                  % Input Modulation alphabet size
k = log2(M);                             % Bits/symbol
%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRANSMITTER %%%%%%%%%%%%
FrmLen = 1000 ;                         % Frame length in information bits
data =   randi([0 M-1],FrmLen,1);        % generate random information bits and write it into the vector data
SymbSet = 0:M-1;                         % Define modulation symbol order, eg., for M=2 the symbols in the constellation set are 0,1
const = pskmod(SymbSet,M);               % Maps bits to symbols in the constellation set using phase shift keying
Scale = modnorm(const,'avpow',1);
dataMod = Scale*pskmod(data,M);          % Scale information symbols to have unit evergy or normalised average power
delay =  0;%input('delay in number of samples' ); % Introduce delay (samples) to model the propagation delay
timeOffset = zeros(delay,1);
Samples_per_symbol = 10;  

Pulshaping_type = 2;%input('Enter 1 to choose Rectangular Pulse shaping OR Enter 2 to choose RRC pulse types') %#ok<NOPTS>

if Pulshaping_type == 1
    disp('Rectangular Pulse type is chosen')
dataMod_upsampled = upsample(dataMod,Samples_per_symbol);
Rectangular_pulse = ones(1,Samples_per_symbol);
txSig = conv(dataMod_upsampled,Rectangular_pulse);
           % timeoffset in samples
txSig = txSig(1:end-Samples_per_symbol+1);

txSig = [timeOffset; txSig(1:end-delay)];

elseif Pulshaping_type == 2                        % Raised Cosine pulse shaping
FilterSpan = 14;                                   % Filter FilterSpan 
rolloff_transmitter =0.8;                         % Rolloff factor
rolloff_receiver  = 0.2;                           % rolloff_receiver = rolloff_transmitter to implement a matched filter

FiltCoeff_transmitter = rcosdesign(rolloff_transmitter,FilterSpan,Samples_per_symbol,'sqrt');
FiltCoeff_receiver = rcosdesign(rolloff_receiver,FilterSpan,Samples_per_symbol,'sqrt');

txSig = upfirdn(dataMod,FiltCoeff_transmitter,Samples_per_symbol); % Filter the modulated data
txSig = [timeOffset; txSig(1:end-delay)];
fprintf('Raised cosine Pulse type chosen with \n Filter span = %d\n Transmitter roll off =%s\n Receiver rolloff = %s\n',FilterSpan,rolloff_transmitter,rolloff_receiver)

else
    display('Input either Rectangular OR RRC pulse types')
end



%%%%%%%%%%%%%% AWGN CHANNEL %%%%%%%%%%%%%%%%%%%%%%%%
EsNovec =SNR;   % input('signal to noise ratio per bit ') Choose the operating SNR per bit

for snr_iter = 1:length(EsNovec)
EbNo = EsNovec(snr_iter);
% Calculate the SNR for an oversampled QPSK signal.
snr = EbNo +10*log10(k)-10*log10(Samples_per_symbol); % calculate the SNR per sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RECEIVER %%%%%%%%%%%%%%%%%%%%%%%%
% Add AWGN
rxSig = awgn(txSig,snr,'measured');
if Pulshaping_type == 1
rxSigFilt = intdump(rxSig, Samples_per_symbol);% Apply integrate and dump receive filter and downsample to symbol rate
% eyediagram(real(txSig(FilterSpan*Samples_per_symbol+1:end-FilterSpan*Samples_per_symbol)),2*Samples_per_symbol) ;
% title('Transmit signal eye diagram')
eyediagram(real(rxSigFilt),2*Samples_per_symbol) ;
 title('Receive signal eye diagram')
elseif Pulshaping_type == 2
rxSigFilt = upfirdn(rxSig, FiltCoeff_receiver,1,Samples_per_symbol);% Apply the RRC receive matched filter and downsample to symbol rate
rxSigFilt = rxSigFilt(FilterSpan+1:end-FilterSpan); % Remove symbols correpsonding to filter span
eyediagram(real(txSig(FilterSpan*Samples_per_symbol+1:end-FilterSpan*Samples_per_symbol)),2*Samples_per_symbol) ;
title('Eye diagram of transmit signal after pulse shaping')
eyediagram(real(rxSig(FilterSpan*Samples_per_symbol+1:end-FilterSpan*Samples_per_symbol)),2*Samples_per_symbol) ;
title('Eye diagram of received signal ')
eyediagram(real(rxSigFilt(FilterSpan*Samples_per_symbol+1:end-FilterSpan*Samples_per_symbol)),2*Samples_per_symbol) ;
title('Eye diagram of received signal after receiver filtering')
end
   
dataOut = pskdemod(rxSigFilt,M);
Errors = sum(dataOut ~= data);
BER_simulation(snr_iter) = Errors/(FrmLen);
end

display('SNR  BER')

for snr_iter = 1:length(EsNovec)
SNR = EsNovec(snr_iter);
fprintf(num2str([SNR BER_simulation(snr_iter)]));
display(' ')
end
%fprintf(' The time offset in samples equal to %d\n',delay);

