function H = Ray_model(L)
% from MIMO-OFDM_Wireless_Communications_with_MATLAB
% Input : L = Number of channel realizations
% Output: H = Channel vector
H = (randn(1,L) +1i*randn(1,L))/sqrt(2);