close all
clc
%% Begin Time Invariant Model.
addpath('functions');

% Defining some useful variables
FFTLength = 2^10;
SymbolsPerFrame = 11;
BitsPerSymbol = 2;
% muVec = [0.05, 0.1, 0.5, 1, 1.5];
muVec = [0.01, 0.05, 0.1, 0.2, 0.35];
% muVec = [2.5];
frameSize = FFTLength*SymbolsPerFrame*BitsPerSymbol;
C = physconst('light');

maxJE = 0;
maxJ = 0;
maxD = 0;

% Defining some figures for later use
figureA = figure(1);
figureB = figure(2);
figureC = figure(3);

% Defining equalisation variables
FilterLength = SymbolsPerFrame;
e = zeros(FFTLength,SymbolsPerFrame);
WienerSolution = zeros(FFTLength,FilterLength);
WienerError = zeros(FFTLength,SymbolsPerFrame);

%% Defining all the channel model components.
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
snr = 10 + 10*log10(2);

%% Generate channel response
% Random channel all sub-carriers experience the same 
% fading so I can measure ensemble averaging.
w = (1+1i)+(randn + 1i*randn);
% W = randn(FFTLength, 1) + 1i.*randn(FFTLength,1);
for i = 1:FFTLength
    W(i,1) = w;
end
%% Channel model
for i = 1:length(muVec)
    J_vec = [];
    D_vec = [];
    JE_vec = [];
    
    x1 = newRandomBinaryFrame(frameSize);
    qpskTx = reshape(QPSKmod(x1), [FFTLength, SymbolsPerFrame]);
    desiredOutput = qpskTx;
    % modulate without channel impairments to evaluate noise.
    qpskOFDMTx = ofdmQpskMod(qpskTx);
    % Apply channel to frame
    for j = 1:SymbolsPerFrame
       qpskTx(:,j) = W.*qpskTx(:,j);
    end
    powerdB = 10.*log10(var(qpskOFDMTx));
    noiseVar = 10^(0.1*(powerdB-snr));
    % remodulate the OFDM now with channel impairments
    qpskOFDMTx = ofdmQpskMod(qpskTx);
    qpskOFDMTx = awgnChannel(qpskOFDMTx, noiseVar);

    % Equaliser
    equalisedOFDM = ofdm4QAMDemod(qpskOFDMTx);
    H_hat = zeros(FFTLength, 1);
    lambda = zeros(FFTLength,1);
    for j = 1:FFTLength
      lambda(j) = var(equalisedOFDM(j,:));
    end
    lambda_med = median(lambda);
    lambda_min = min(lambda);
    lambda_max = max(lambda);
    lambda_mean = mean(lambda);
    S = 2/lambda_max;
    clear lambda; % clean up some memory.

    % Evaluate the Wiener Solution
    if(i == 1)
       for j = 1:FFTLength
           [~, WienerSolution(j,:)] = ...
               One_Tap_Wiener_Filter(equalisedOFDM(j,1:FilterLength), ...
               desiredOutput(j,1:FilterLength), ...
               qpskTx(j,1:FilterLength));
       end
       WienerSolution = mean(WienerSolution,2);  
    end
    mu = muVec(i);
    % The time taking roughly for the slowest converging coefficient 
    % to converge is v_k(0)=v_k(0)(1-\mu\lambda_k)^n assuming the error 
    % is 2 to begin with (which given the random channel coefficient is 
    % from a gaussian distribution should cover most cases. n for 
    % convergence to within 10^-3 can be approximated as
    n_max = -3/(log10(1-1*mu*lambda_min));
    for j = 1:max(floor(n_max/SymbolsPerFrame),1)
        for k = 1:SymbolsPerFrame
            % mean square deviation
            D = mean(abs(conj(WienerSolution)-conj(H_hat)).^2);
            
            
            e(:,k) = desiredOutput(:,k) - conj(H_hat).*equalisedOFDM(:,k);
%             H_hat = H_hat + (mu./(0.05+abs(equalisedOFDM(:,k)).^2)).*equalisedOFDM(:,k).*conj(e(:,k));
            H_hat = H_hat + mu.*equalisedOFDM(:,k).*conj(e(:,k));
            WienerError(:,k) = desiredOutput(:,k) - conj(WienerSolution)...
                .*equalisedOFDM(:,k);
            Wiener_minimum_mean_square_error = var(WienerError(:,k));
            % mean square error
            J = mean(abs(e(:,k)).^2);
            J_Excess = J - Wiener_minimum_mean_square_error;
            JE_vec = [JE_vec, J_Excess];

            minimum_mean_square_weight_error = mu.*...
                Wiener_minimum_mean_square_error./(2-mu*lambda_mean) + ...
                (1-mu*lambda_mean).^(2*max(n_max,SymbolsPerFrame))...
                .*(abs(1./W(1)).^2 - ...
                (mu*Wiener_minimum_mean_square_error./(2-mu*lambda_mean)));

            J_vec = [J_vec, J];
            D_vec = [D_vec, D];
           
        end
       
    figure(figureA);
    hold on
    plot(J_vec);
    title("Mean square error vs training symbols");
    ylabel("Mean square error");
    xlabel("training symbols");
    if(maxJ < max(J_vec)); maxJ = max(J_vec); else; maxJ=maxJ; end
    axis([0,200,0,maxJ]);
    figure(figureB);
    hold on
    plot(D_vec);
    title("Mean square deviation vs training symbols");
    ylabel("Mean square deviation");
    xlabel("training symbols");
    if(maxD < max(D_vec)); maxD = max(D_vec); else; maxD=maxD; end
    axis([0,200,0,maxD]);
    figure(figureC);
    plot(JE_vec);
    hold on
    title("Excess mean square error vs training symbols");
    ylabel("Excess mean square error");
    xlabel("training symbols");   
    if(maxJE < max(JE_vec)); maxJE = max(JE_vec); else; maxJE=maxJE; end
    axis([0,200,0,maxJE]);
    end
end
string1 = "mu = " + muVec(1);
string2 = "mu = " + muVec(2);
string3 = "mu = " + muVec(3);
string4 = "mu = " + muVec(4);
string5 = "mu = " + muVec(5);
figure(figureA);
% legend(string1);
legend(string1, string2, string3, string4, string5);
figure(figureB);
% legend(string1);
legend(string1, string2, string3, string4, string5);
figure(figureC);
% legend(string1);
legend(string1, string2, string3, string4, string5);
