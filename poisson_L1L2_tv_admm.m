%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs L1/L2-TV regularized Poisson denoising using ADMM
% algorithm.
% Input:
%   f: image to be denoised and deblurred
%   A: blurring operator
%   beta: parameter for the fidelity term. Larger if solution needs to be
%         closer to the image f
%   rho: penalty parameter for the constraint Du = g
%   P: peak value of image; upper bound pixel intensity of the solution u.
% Output:
%   u: deblurred and denoised image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = poisson_L1L2_tv_admm(f, A, beta, rho, P)


[m, n] = size(f);

% initialization
u = zeros(m,n);
t1_dual = zeros(m, n); 
t2_dual = zeros(m, n);

v = zeros(m, n);
q = f;
y = zeros(m, n); % dual for v
w1 = zeros(m, n); 
w2 = zeros(m, n); % duals for d1, d2
z = zeros(m, n); % dual for q

tol = 1e-5;

mu = 1;
gamma = 1;
lambda = 1;
J = 10;

% fft_blur
rows = m;
cols = n;
%refit blurring operator and shift it
[xLen_flt, yLen_flt] = size(A);
ope_blur=zeros(rows,cols);
ope_blur(1:xLen_flt,1:yLen_flt)=A;
    
xLen_flt_1=floor(xLen_flt/2);yLen_flt_1=floor(yLen_flt/2);
ope_blur_1=padarray(ope_blur,[rows,cols],'circular','pre');
ope_blur_1=ope_blur_1(xLen_flt_1+1:rows+xLen_flt_1,yLen_flt_1+1:cols+yLen_flt_1);
    
%fourier transform of blurring operator
FA = fft2(ope_blur_1); 

% FFT-pre for u_{j+1}
eigx = fft2([1, -1], m, n);
eigy = fft2([1; -1], m, n);
eigDx = abs(eigx).^2;
eigDy = abs(eigy).^2;
FA2 = abs(FA).^2; 
Denom = lambda*FA2 + (rho + gamma) * (eigDx + eigDy) + mu;


for k = 1:300
    u_prev = u;

    % g-update
    Du1 = Dx(u); 
    Du2 = Dy(u);
    G1 = Du1 + t1_dual;
    G2 = Du2 + t2_dual;
    % L1,L2 norms
    Du_norm1 = sum(abs(Du1(:))) + sum(abs(Du2(:)));       
    G_norm2 = sqrt(sum(G1(:).^2) + sum(G2(:).^2)); 

    if G_norm2 < tol
        g1 = (Du_norm1/rho)^(1/3)*ones(size(G1))/sqrt(numel(G1)*2);
        g2 = (Du_norm1/rho)^(1/3)*ones(size(G2))/sqrt(numel(G2)*2);
    else
        kappa = Du_norm1 / (rho * G_norm2^3);
        zeta = ((27 * kappa + 2) + sqrt((27*kappa+2)^2 - 4))/2;
        zeta = zeta^(1/3);
        tau = (1/3) + (1/3)*(zeta + 1/zeta);
        g1 = tau*G1;
        g2 = tau*G2;
    end

    % u-update
    % initial
    
    d1 = Du1; 
    d2 = Du2;
    gt1 = g1 - t1_dual;
    gt2 = g2 - t2_dual;
    Dtg = Dxt(gt1) + Dyt(gt2);  



    % ||g^(k)||_2 for d_{j+1}
    g_norm2 = sqrt(sum(g1(:).^2) + sum(g2(:).^2));  % L2 norm of current gradient
    nu = 1 / (gamma * g_norm2);  % shrinkage threshold

    for j = 1:J
        % c_j+1
        dw1 = d1 - w1;
        dw2 = d2 - w2;
        Dtdw = Dxt(dw1) + Dyt(dw2);

        % A^T(q-z)
        qz = fft2(q-z);
        ATqz = real(ifft2(conj(FA) .*qz));
 
        c = rho * Dtg + lambda * ATqz + gamma * Dtdw + mu * (v - y);

        % u_{j+1} by FFT      
        c_fft = fft2(c);
        u = ifft2(c_fft ./ Denom);

        % q_{j+1}
        Au = real( ifft2( FA .* fft2(u) ) );
        tmp = lambda * Au + lambda * z - beta;  
        q = (tmp + sqrt(tmp.^2 + 4 * beta * lambda * f)) ./ (2 * lambda); 

        % d_{j+1} 
        Du1 = Dx(u);  % Gradient in x direction
        Du2 = Dy(u);  % Gradient in y direction
    
        d1 = sign(Du1 + w1) .* max(abs(Du1 + w1) - nu, 0);
        d2 = sign(Du2 + w2) .* max(abs(Du2 + w2) - nu, 0);

        % v_{j+1}
        v = max(min(u + y, P), 0);

        % dual-update
        y = y + u - v;
        w1 = w1 + Du1 - d1;
        w2 = w2 + Du2 - d2;
        z = z + Au - q;
    end

    % t-update
    t1_dual = t1_dual + Dx(u) - g1;
    t2_dual = t2_dual + Dy(u) - g2;

    % stopping criterion
    if norm(u - u_prev, 'fro')/norm(u_prev, 'fro') < tol
        break;
    end

end
end






   

