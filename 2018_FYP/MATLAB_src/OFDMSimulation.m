clear all;
clc;
%% Begin code here
ofdmMod = comm.OFDMModulator( ...
    'FFTLength',            256, ...
    'NumGuardBandCarriers', [0;0], ...
    'InsertDCNull',         false, ...
    'PilotInputPort',       false, ...
    'CyclicPrefixLength',   0, ...
    'NumSymbols',           10, ...
    'NumTransmitAntennas',  2);

% Generate an OFDM demodulator based on the modulation
ofdmDemod = comm.OFDMDemodulator(ofdmMod);

% Create random sequence of [01]*
modIn = randi([0 1], 256,10, 2);
% Create time domain sequence from modIn frequency domain signal
modOut = ofdmMod(modIn);

% Demodulate signal to Frequency domain.
receiveOutCmplx=ofdmDemod(modOut(:,1));

%% compare signals they are the same.
receiveOut=real(receiveOutCmplx);
% Arbitrarily set some threshold to be equal to 0
threshold = 0.0001;

receiveOut(receiveOut<threshold)=0;
difference = receiveOut - modIn(:,:,1);
difference(difference<threshold)=0;
difference(difference~=0)=1;

% check for 1's in the matrix.
if ~ismember(1, difference)
    fprintf("Signal Input == Signal Output\n");
else
    fprintf("Signal Input ~= Signal Output\n");
end
    

