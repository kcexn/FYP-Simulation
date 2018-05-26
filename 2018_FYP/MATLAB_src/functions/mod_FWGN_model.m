function [h, Nfft, Nifft, doppler_coeff] = mod_FWGN_model(fm, fs, N, ps, nLength, delta_t, Ts)
% Input: fm = Maximum Doppler frequency
%        fs = Sampling frequency, N = number of samples
% Output: h = Complex fading channel

mLength = nLength;
R = zeros(nLength);
for n = 1:nLength
    for m = 1:mLength
        for k = 1:length(ps)
           R(n,m) = R(n,m) + ps(k).*sinc(delta_t(k)./Ts - m).*sinc(delta_t(k)./Ts -n).*Ts/1000;
        end
    end
end
L = chol(R);

Nfft = 2^max(3,nextpow2(2*fm/fs*N));
% fprintf("nextpow2 is: %d", nextpow2(2*fm/fs*N));
Nifft = ceil(Nfft*fs/(2*fm));

% Generate the independent complex Gaussian random process
GI = randn(nLength,Nfft); GQ = randn(nLength,Nfft);
% Take FFT of real signal in order to make hermitian symmetric
CGI = fft(GI,[],2); CGQ = fft(GQ,[],2);
% Nfft sample Dopppler spectrum generation
doppler_coeff = Doppler_spectrum(fm, Nfft);
% Do the filtering of the Gaussian random variables here
f_CGI = CGI.*sqrt(doppler_coeff); f_CGQ = CGQ.*sqrt(doppler_coeff);
% Adjust sample size to take IFFT by (Nifft-Nfft) sample zero padding
Filtered_CGI = [f_CGI(:, 1:Nfft/2) zeros(nLength,Nifft-Nfft) f_CGI(:,Nfft/2+1:Nfft)];
Filtered_CGQ = [f_CGQ(:, 1:Nfft/2) zeros(nLength, Nifft-Nfft) f_CGQ(:,Nfft/2+1:Nfft)];
hI = ifft(Filtered_CGI,[],2); hQ = ifft(Filtered_CGQ,[],2);
% Take the magnitude squared of the I and Q components and add them
rayEnvelope = sqrt(abs(hI).^2 + abs(hQ).^2);
% Compute the root mean squared value and normalize the envelope
rayRMS = sqrt(mean(rayEnvelope(1:N).*rayEnvelope(1:N)));
Z = complex(real(hI(:, 1:N)), - real(hQ(:, 1:N)))/rayRMS;
for i = 1:N
    h(:,i) = L * Z(:,i);
end
