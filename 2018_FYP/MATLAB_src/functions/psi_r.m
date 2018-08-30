function [psi_r] = psi_r(t,Ts,tau)

psi_r = sqrt(2/Ts).*sinc((2/Ts).*((t+tau)-0.1*Ts));