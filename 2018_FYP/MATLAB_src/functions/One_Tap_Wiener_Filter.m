function [Rx_f, W] = One_Tap_Wiener_Filter(Rx, Tx, Noiseless_Rx, noiseVar)
% Solves the Wiener Solution channel estimate for a given number of inputs
% Filters the input Rx based on the Noiseless_Rx and Tx.
% Finds W based on W = R^(-1)*P. A Pseudo Inverse is used since the Inverse
% may not always be numerically stable.
%
% Returns the filtered solution for the input vector.
%
% Tx, Rx and Noiseless_Rx all MUST be Vectors.

% begin the function
% Rx_f = zeros(size(Rx));
% R = mean(Rx.*conj(Rx));
R = mean(Noiseless_Rx.*conj(Noiseless_Rx))+ noiseVar;
P = Noiseless_Rx.*conj(Tx);
W = inv(R).*P;
Rx_f = conj(W).*Rx;


% 
% 
% for i = 1:size(Rx,1)
% 
%     % Can use this estimate because the length one random process is
%     % gaussian centred about a mean set by the Rayleigh fading channel.
%     % Hence it's ensemble average should reduce to the noiseless 
%     % equivalent.
% %     R = Noiseless_Rx(i).*conj(Noiseless_Rx(i));
%     P = Noiseless_Rx(i).*conj(Tx(i));
%     W = inv(R)*P;
%     Rx_f(i) = conj(W).'*Rx(i);
% end

