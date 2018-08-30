function [Rx_f] = Wiener_Filter(Rx, Tx)
% Solves the Wiener Solution channel estimate for a given number of inputs
% Filters the input Rx based on the Noiseless_Rx and Tx.
% Finds W based on W = R^(-1)*P. A Pseudo Inverse is used since the Inverse
% may not always be numerically stable.
%
% Returns the filtered solution for the input vector.
%
% Tx, Rx and Noiseless_Rx all MUST be Vectors.

if(~isvector(Rx) || ~isvector(Tx) )
    error('Inputs must ALL be vectors.');
    return;
end

% Make all vectors column vectors
if( size(Rx,2) > size(Rx,1) )
    Rx = Rx.';
end

if ( size(Tx,2) > size(Tx,1) )
    Tx = Tx.';
end

% Check to make sure all vectors are of the same size
num_check = {'numeric'};
attributes = {'size', size(Rx)};
validateattributes(Rx, num_check, attributes);
validateattributes(Tx, num_check, attributes);

% begin the function
Rx_f = zeros(size(Rx));
for i = 1:size(Rx,1)
    U = Rx(i);
%     U = Noiseless_Rx(i:-1:1);
    R = xcorr(U);
    R = toeplitz(R);
    P = U.*conj(Tx(i));
    W = inv(R)*P;
    Rx_f(i) = conj(W).'*U;
end

