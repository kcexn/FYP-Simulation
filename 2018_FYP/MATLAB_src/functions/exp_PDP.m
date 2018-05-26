function PDP = exp_PDP(tau_d, Ts, A_dB, norm_flag)
% from MIMO-OFDM_Wireless_Communications_with_MATLAB
% Exponential PDP generator
%   Inputs:
%     tau_d: rms delay spread[sec]
%     Ts: Sampling time[sec]
%     A-dB: smallest noticeable power[dB]
%     norm_flag: normalize total power to unit
%   Output:
%     PDP: PDP vector
if nargin<4, norm_flag=1; end %normalization
if nargin<3, A_dB=-20; end
sigma_tau = tau_d; A=10^(A_dB/10);
lmax=ceil(-tau_d*log(A)/Ts);
% compute normalization factor for power normalization
if norm_flag
    p0 = (1 - exp(-Ts/sigma_tau))/(1-exp((lmax+1)*(-Ts)/sigma_tau)); 
else
    p0 = 1/sigma_tau;
end
% Exponential PDP
l = 0:lmax; PDP = p0*exp(-l*Ts/sigma_tau);