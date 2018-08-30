function [psi_s] = psi_s(t,Ts,tau)

psi_s = sqrt(2/Ts).*sinc((2/Ts).*(t-tau));