close all
clear
clc
%% Code Starts Here:

% Defining some useful arrays
EbNoVec = (0:13)';
berTheoryVecQPSK= [];
berVecQPSK = [];

berVec16QAM = [];
berTheoryVec16QAM = [];

for m=1:length(EbNoVec)
    %% QPSK Modulation, OFDM channel.
    % pnSequence
    pnSequenceQPSK = comm.PNSequence( ...
        'Polynomial', [12 6 4 0], ...
        'SamplesPerFrame', 3968, ...
        'InitialConditions', [1 0 0 0 1 0 1 0 0 1 0 1]);
    x1 = pnSequenceQPSK();
    
    % QPSK modulation
    QPSKmod = comm.QPSKModulator( ...
        'BitInput', true ...        
        );
    qpskTx = QPSKmod(x1);
    
    qpskTx = reshape(qpskTx, [64 31]);
    
    % OFDM Modulation
    ofdmQpskMod = comm.OFDMModulator( ...
        'FFTLength',            64, ...
        'NumGuardBandCarriers', [0;0], ...
        'InsertDCNull',         false, ...
        'PilotInputPort',       false, ...
        'CyclicPrefixLength',   0, ...
        'NumSymbols',           31, ...
        'NumTransmitAntennas',  1);

    qpskTx = ofdmQpskMod(qpskTx);
    
    % AWGN Channel
    % SNR in dB
    EbNo = EbNoVec(m);
    % SNR taking into account 2 bits per QAM symbol?

    % The addition of the 10*log10(2) comes straight out of the
    % definition of SNR in relation to EbN0
    % Eb/N0 = (S/N)(W/Rb)
    snr = EbNo + 10*log10(2);

    powerDB = 10*log10(var(qpskTx));
    noiseVar = 10.^(0.1*(powerDB-snr));
    
    awgnChannel = comm.AWGNChannel( ...
        'NoiseMethod', 'Variance', ...
        'VarianceSource', 'Input port' ...
        );
    qpskTx = awgnChannel(qpskTx, noiseVar);
    
    % OFDM Demodulator
    ofdm4QAMDemod = comm.OFDMDemodulator(ofdmQpskMod);
    qpskRx = ofdm4QAMDemod(qpskTx);
    
    qpskRx = reshape(qpskRx, [1984, 1]);
    
    %QPSK Demodulator
    qpskDemod = comm.QPSKDemodulator( ...
        'BitOutput',true ...
        );
    dataOut = qpskDemod(qpskRx);
    
    berVecQPSK = [berVecQPSK, sum(x1 ~= dataOut)/length(dataOut)];
    berTheoryVecQPSK = [berTheoryVecQPSK, berawgn(EbNo, 'psk', 4, 'nondiff')];
    
    %% 16-QAM Modulation
    % pnSequence
    pnSequence16QAM = comm.PNSequence( ...
        'Polynomial', [12 6 4 0], ...
        'SamplesPerFrame', 3840, ...
        'InitialConditions', [1 0 0 0 1 0 1 0 0 1 0 1]);
    x1 = pnSequence16QAM();
    
    % 16QAM Modulator
    tx16QAM = qammod(x1, 16, 'gray', ...
        'InputType', 'bit' ...
    );
    
    % OFDM Modulation
    tx16QAM = reshape(tx16QAM, [64 15]);
    
    ofdm16QAMMod = comm.OFDMModulator( ...
        'FFTLength',            64, ...
        'NumGuardBandCarriers', [0;0], ...
        'InsertDCNull',         false, ...
        'PilotInputPort',       false, ...
        'CyclicPrefixLength',   0, ...
        'NumSymbols',           15, ...
        'NumTransmitAntennas',  1);

    tx16QAM = ofdm16QAMMod(tx16QAM);
    
    % AWGN Channel
    snr = EbNo + 10*log10(4);
    powerDB = 10*log10(var(tx16QAM));
    noiseVar = 10.^(0.1*(powerDB-snr));
    
    tx16QAM = awgnChannel(tx16QAM, noiseVar);
    
    % OFDM Demodulator
    ofdm16QAMDemod = comm.OFDMDemodulator(ofdm16QAMMod);
    rx16QAM = ofdm16QAMDemod(tx16QAM);
    
    rx16QAM = reshape(rx16QAM, [960, 1]);
    
    % 16QAM demodulator
    dataOut = qamdemod( ...
        rx16QAM, 16, ...
        'OutputType', 'bit' ...
        );
    
    % Bit error rate
    berVec16QAM = [berVec16QAM, sum(dataOut ~= x1)/length(dataOut)];
    berTheoryVec16QAM = [berTheoryVec16QAM, berawgn(EbNo, 'qam', 16, 'nondiff')];
    
end

figure()
semilogy(EbNoVec, berVecQPSK, 'r*');
hold
semilogy(EbNoVec, berTheoryVecQPSK);
semilogy(EbNoVec, berTheoryVec16QAM);
semilogy(EbNoVec, berVec16QAM, 'g*');
xlabel("E_b/N_0");
ylabel("BER");
legend("QPSK Simulated", "QPSK Theory", "16 QAM Theory", "16 QAM Simulated");
title("BER vs E_b/N_0 for QPSK and 16 QAM modulation");