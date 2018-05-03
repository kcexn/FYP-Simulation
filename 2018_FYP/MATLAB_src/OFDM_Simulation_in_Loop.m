clear
clc
%% Code starts here:
% Need to make some decisions about symbol rate.
% Search term: OFDM TUtorial

% Unable to use tuples I've taken to using a 2 dimensional
% matrix where the first row represents the real values
% and the second row represents the imaginary component
% of a given complex number.
valueArr = [[1; 1],[-1; 1],[-1; -1],[1; -1]];
bitKeys={'00','01','11','10'};
valueSet=[1, 2, 3, 4];
bitDictionary = containers.Map(bitKeys,valueSet);

% Defining some useful arrays
EbNoVec = (1:8)';
berTheoryVec = [];
berVec = [];

for m = 1:length(EbNoVec)
    %% Generate a pseudo random binary sequence ( y )
    pnSequence = comm.PNSequence( ...
        'Polynomial', [12 6 4 0], ...
        'SamplesPerFrame', 4095, ...
        'InitialConditions', [1 0 0 0 1 0 1 0 0 1 0 1]);
    x1 = pnSequence();

    %% Map pnSequence to an NxM matrix of complex digital values
    % Start with 4-QAM

    sym4QAM = zeros(64, 1);
    tx4QAMSym = [];
    txSym = [];
    % Map the PN sequence to an NxM matrix here.
    % Where N is the number of subcarriers
    % and M is the number of symbols.

    % 4QAM Symbol generator.
    for i = 1:128:3968
        for j = 1:64
           % Yep, highly readable indexing into the pseudorandom number sequence.
           % Blame the extra unnecessary subtraction on 1 indexing...
           key=strcat(int2str(x1(2*j+i-2)),int2str(x1(2*j+i-1)));
           sym4QAM(j)=complex(valueArr(1,bitDictionary(key)),valueArr(2,bitDictionary(key)));
        end
        tx4QAMSym = [tx4QAMSym, sym4QAM];
    end

    % OOK Symbol Generator
    for i = 1:64:4032
        txSym = [txSym, x1(i:i+63)];
    end

    %% OFDM Modulate ( y )

    % 4QAM OFDM Modulator
    ofdm4QAMMod = comm.OFDMModulator( ...
        'FFTLength',            64, ...
        'NumGuardBandCarriers', [0;0], ...
        'InsertDCNull',         false, ...
        'PilotInputPort',       false, ...
        'CyclicPrefixLength',   0, ...
        'NumSymbols',           31, ...
        'NumTransmitAntennas',  1);

    channel4QAMIn = ofdm4QAMMod(tx4QAMSym);
    % plot([1:128], channel4QAMIn(1:128));

    % OOK OFDM Modulator
    ofdmMod = comm.OFDMModulator( ...
        'FFTLength',            64, ...
        'NumGuardBandCarriers', [0;0], ...
        'InsertDCNull',         false, ...
        'PilotInputPort',       false, ...
        'CyclicPrefixLength',   0, ...
        'NumSymbols',           63, ...
        'NumTransmitAntennas',  1);

    channelIn = ofdmMod(txSym);

    %% Add additive White Gaussian noise ( y )

    % SNR in dB
    EbNo = EbNoVec(m);
    % SNR taking into account 2 bits per QAM symbol?

    % The addition of the 10*log10(2) comes straight out of the
    % definition of SNR in relation to EbN0
    % Eb/N0 = (S/N)(W/Rb)
    snr = EbNo + 10*log10(2);

    powerDB = 10*log10(var(channel4QAMIn));
    noiseVar = 10.^(0.1*(powerDB-snr));

    awgn4QAMChan = comm.AWGNChannel('NoiseMethod', 'Variance', ...
        'VarianceSource', 'Input port');

    channel4QAMIn = awgn4QAMChan(channel4QAMIn, noiseVar);

    %% OFDM Demodulate ( y ) 
    %4-QAM Demodulator
    ofdm4QAMDemod = comm.OFDMDemodulator(ofdm4QAMMod);
    channel4QAMOut = ofdm4QAMDemod(channel4QAMIn);

    % OOK Demodulator
    ofdmDemod = comm.OFDMDemodulator(ofdmMod);
    channelOut = ofdmDemod(channelIn);


    %% Map output complex baseband digital values to binary sequence.
    % Once I add noise I'll have to make the decisions here
    % Based off of Euclidean distance.

    %4QAM rxSym
    %I'm going to map decisions based on quadrant
    rx4QAMSym = [];
    for i = 1:size(channel4QAMOut,2)
        for j = 1:size(channel4QAMOut,1)
    %         sign(real(channel4QAMOut(j,i)))
            reSym = sign(real(channel4QAMOut(j,i)));
            imSym = sign(imag(channel4QAMOut(j,i)));
            demodKey=strcat(int2str(reSym),int2str(imSym));
            switch(demodKey)
                case '11'
                    rx4QAMSym=[rx4QAMSym; [0;0]];
                case '-11'
                    rx4QAMSym=[rx4QAMSym; [0;1]];
                case '-1-1'
                    rx4QAMSym=[rx4QAMSym; [1;1]];
                case '1-1'
                    rx4QAMSym=[rx4QAMSym; [1;0]];
                otherwise
                    fprintf('ERROR!\n')
                    % Return early due to error
                    return
            end
        end
    end


    % OOK rxSym
    rxSym = [];
    % The size here should be retrieved from the matrix.
    for i = 1:63
        rxSym = [rxSym; channelOut(:,i)]; 
    end

    %% compare binary sequences and count bit errors.
    % Match BER to Theoretical calculation.

    bitErrors = sum(x1(1:3968) ~= rx4QAMSym);
    berVec = [berVec, bitErrors/length(rx4QAMSym)];
    % Chapter 4 of Bernard Sklar's textbook on communications
    % theory has some excellent error curves for MPSK modulation.
    berTheoryVec = [berTheoryVec, berawgn(EbNo, 'psk', 4, 'nondiff')];

%     isequal(x1(1:3968), rx4QAMSym);
%     isequal(x1(1:4032), uint8(real(rxSym)));
end

figure()
semilogy(EbNoVec, berVec, 'rx');
hold
semilogy(EbNoVec, berTheoryVec);
xlabel("E_b/N_0");
ylabel("BER");
legend("Simulated", "Theory");
title("BER vs E_b/N_0");
