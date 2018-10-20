clear all
SNR_vec = [0:4:30];KVEC=[0 10 100]
for k_loop = 1:length(KVEC)
    K = KVEC(k_loop);
for realizations = 1:1000
for snr_loop = 1:length(SNR_vec)
    snr_instant = 10^(SNR_vec(snr_loop)/10);
    temp_rand  = sqrt(0.5)*(randn(1,1)+j*randn(1,1));
    Hrician = sqrt(K/(K+1))*  temp_rand  + sqrt(1/(K+1))* temp_rand ;
    Hrician_instant =  Hrician;
    C_iid(snr_loop) = sum(log2(1 +( snr_instant*temp_rand*temp_rand')));
    C_rician(snr_loop) = sum(log2(1 +( snr_instant* Hrician_instant* Hrician_instant')));
end
C_iid_reali(:,realizations)= C_iid;
C_rician_reali(:,realizations)= C_rician;
end
C_iid_reali=C_iid_reali;
plot(SNR_vec,mean(C_rician_reali,2))
temp(:,k_loop) =abs(mean(C_rician_reali,2));
hold on
end
legend({'K= 0','K= 10','K= 100'})
plot(SNR_vec,mean(C_iid_reali,2),'--')
hold on