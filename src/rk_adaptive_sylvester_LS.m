function [Xu, Xv,k,nrmrestotnew,RES] = rk_adaptive_sylvester_LS(A, Bdiag, u, options)
% [Xu, Xv,k,nrmrestotnew,RES] = rk_adaptive_sylvester_LS(A, Bdiag, u, options)
%
% Rational Krylov method for Sylvester equations that ensures the last pole
% is equal to infinity at each step, to check the residual.
%
% The function solves the Sylvester equation
%
%    A X - X diag(Bdiag) - u ones(length(Bdiag),1)' = 0
%
%
%Input parameters:
%  A square matrix of dimension nA x nA 
%  Bdiag vector of length nB
%  u  vector of length nA 
%
%  options.maxit = max space dimension, defalut min(sqrt(size(A)),sqrt(size(B)));
%
%  options.tol = max final accuracy (in terms of relative residual),
%  default 1e-10.
%
%  options.poles determines how to adaptively chose poles. The possible
%  choices are options.poles='ADM', options.poles='sADM', default is 'sADM'
%
%  The pole selection algorithm are described in detail in [1]. Algorithm
%  'ADM' is the heuristic proposed in [2].
%
%
%  options.mA, option.MA determines a lower and upper bound, respectively, for
%  the real part of the eigenvalues of A;
%
%  options.mB, option.MB determines a lower and upper bound, respectively, for
%  the real part of thr eigenvalues of B;
%
%  options.real='true' runs the algorithm for A and B real;
%
% Output parameters:
%
% Xu,Xv = solution factors   X_approx = Xu Xv'
% nrmrestotnew: residual norm history
% RES: nAxnB matrix whose (i,j) entry is the relative residual norm related
%      related to the j-th shifted linear system at the i-th iteration
%
% This code has been designed by modyfing Casulli and Robol's solver
% for Sylvester equations, see
%
% A. Casulli and L. Robol 
% An Efficient Block Rational Krylov Solver for Sylvester Equations with Adaptive Pole Selection 
% SIAM Journal on Scientific Computing, 46 (2024), pp. A798–A824, https://doi.org/10.1137/23M1548463.
%
%THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
%IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
%FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
%COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
%IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
%CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 


if ~exist('rat_krylov', 'file')
    error('RKTOOLBOX not found, please download it from ', ... 
        'http://guettel.com/rktoolbox/');
end


if isfield(options, 'tol')==0
    options.tol=1e-10;
end

if isfield(options, 'real')==0
    options.real=0;
end


if isfield(options, 'maxit')==0
    options.maxit=min(sqrt(size(A,1)),sqrt(size(Bdiag,1)));
end
maxit=options.maxit;

if isfield(options, 'poles')==0
    options.poles="sADM";
end


[VA,KA,HA] = rat_krylov(A, u, inf);

bs=size(u,2);
C=zeros(maxit+2,bs);
C(1:bs,:)=VA(:,1:bs)'*u;
Ap=HA(1:bs,1:bs)/KA(1:bs,1:bs);

l = length(Bdiag);
Y = zeros((maxit+1)*bs,l);
index_noncov = 1 : l;
nrm_rhs = norm(u,'fro')*ones(1,l);
nrmrestotnew = [];
res = [];

if options.poles=="ext"
    snewA=0;
    sA=snewA;
else    
    if isfield(options, 'MA')==0
        options.MA=eigs(A,1,'SM', 'Maxiterations',1e5);
    end
    MA=options.MA;
    if isfield(options, 'mA')==0
        options.mA=sign(options.MA) * normest(A);
    end
    mA=options.mA;

    if isfield(options, 'mB')==0
        %options.mB=eigs(B,1,'SM', 'Maxiterations',1e5);
        options.mB=-min(abs(Bdiag));
    end
    mB=options.mB;
    if isfield(options, 'MB')==0
        options.MB=-max(abs(Bdiag));
    end
    MB=options.MB;


    if abs(MB)<abs(mB)
        snewA=MB;
    else
        snewA=mB;
    end

    if ~isreal(snewA)
        snewA=[snewA,conj(snewA)];
    end

    if abs(MA)<abs(mA)
        snewB=MA;
    else
        snewB=mA;
    end

    if ~isreal(snewB)
        snewB=[snewB,conj(snewB)];
    end

    sA=[MB+mB-snewA(1),snewA];
end

k=1;
while k<=maxit 
    k=k+1;

    if options.real==1
        [VA,KA,HA,Ap] = Swapped_update_real(A,VA,KA,HA,snewA, Ap);
        k=k+length(snewA)-1;
    else
        [VA,KA,HA,Ap] = Swapped_update(A,VA,KA,HA,snewA, Ap);
    end

  
    %%%% solve the projected least squares problems
    for t = index_noncov
        temp_H = HA + Bdiag(t) * KA;
        [Q,R] = qr(temp_H);
        rhs = Q' * C(1:k+1,:);
        y = R(1:k, 1:k)\rhs(1:k,:);
        res(t) = norm(rhs(k+1,:),'fro');
        Y(1:k+1, t) = KA * y;
        
    end
    
    res=res./nrm_rhs;
    RES(:,k-1)=res;
    index_noncov = find(res>options.tol);
    
    [~,TA]=schur(Ap, 'complex');
    if options.poles=="ADM"
        % these functions are designed for equations of the form
        % AX-XB+uv'=0, that is way we need to change sign to everything
        % that is related to B
        snewA = newpole_adaptive_ADM(-min(abs(Bdiag)),-max(abs(Bdiag)),-Bdiag,diag(TA),sA);
    elseif options.poles=="sADM"
        % these functions are designed for equations of the form
        % AX-XB+uv'=0, that is way we need to change sign to everything
        % that is related to B
        snewA = newpole_adaptive_sADM(-min(abs(Bdiag)),-max(abs(Bdiag)),-Bdiag,diag(TA),sA);
    elseif options.poles=="ext"
        if snewA == inf
            snewA = 0;
        else
            snewA = inf;
        end
    end

    if options.real
        if ~isreal(snewA)
            snewA(2)=conj(snewA);
        end
    end

    sA = [sA, snewA];
    
    %Frobenius norm
    if max(res)<options.tol
        break
    end
end
if k>= maxit
    fprintf('Stopped at iteration %d\n', k);
end
Xu = VA(:,1:end) ;
Xv = Y(1:k+1,:);
end


function [V,K,H,Ap] = Swapped_update(A,V,K,H,s, Ap)
%Step of rational Krylov method to add the pole s
% and swap poles ensuring that that the last pole
% is equal to infinity.

bs = size(H, 1) - size(H, 2);
param.extend=bs;
param.deflation_tol=0;

[V,K,H] = rat_krylov(A,V,K,H,s,param);
k=size(K,2);
l=1;

[Q,R]=qr(K(end-(l+1)*bs+1:end,end-l*bs+1:end));
K(end-(l+1)*bs+1:end,end-l*bs+1:end)=R;
V(:,end-(l+1)*bs+1:end)=V(:,end-(l+1)*bs+1:end)*Q;
H(end-(l+1)*bs+1:end,end-(l+1)*bs+1:end)=...
    Q'*H(end-(l+1)*bs+1:end,end-(l+1)*bs+1:end);
[Q,~]=qr(H(end:-1:end-(l+1)*bs+1,end-(l+1)*bs+1:end)');
Q=Q(:,end:-1:1);
H(:,end-(l+1)*bs+1:end)=H(:,end-(l+1)*bs+1:end)*Q;
K(1:end-bs,end-(l+1)*bs+1:end)=...
    K(1:end-bs,end-(l+1)*bs+1:end)*Q;

e=zeros(k,l*bs);
e(end-l*bs+1:end,:)=eye(l*bs);
Ap(1:end+l*bs,end+1:end+l*bs)=...
    H(1:end-bs,1:end)*(K(1:end-bs,1:end)\e);
Ap(end-l*bs+1:end,1:end)=...
    H(end-(l+1)*bs+1:end-bs,1:end)/K(1:end-bs,1:end);
end


function [V,K,H,Ap] = Swapped_update_real(A,V,K,H,s, Ap)
%Step of rational Krylov method to add the pole s (and its conjugate if s
%is not real), and swap poles ensuring that that the last pole
% is equal to infinity.

bs = size(H, 1) - size(H, 2);
param.extend=bs;
param.deflation_tol=0;
param.real=1;
[V,K,H] = rat_krylov(A,V,K,H,s,param);
k=size(K,2);
l=length(s);

[Q,R]=qr(K(end-(l+1)*bs+1:end,end-l*bs+1:end));
K(end-(l+1)*bs+1:end,end-l*bs+1:end)=R;
V(:,end-(l+1)*bs+1:end)=V(:,end-(l+1)*bs+1:end)*Q;
H(end-(l+1)*bs+1:end,end-(l+1)*bs+1:end)=...
    Q'*H(end-(l+1)*bs+1:end,end-(l+1)*bs+1:end);
[Q,~]=qr(H(end:-1:end-(l+1)*bs+1,end-(l+1)*bs+1:end)');
Q=Q(:,end:-1:1);
H(:,end-(l+1)*bs+1:end)=H(:,end-(l+1)*bs+1:end)*Q;
K(1:end-bs,end-(l+1)*bs+1:end)=...
    K(1:end-bs,end-(l+1)*bs+1:end)*Q;

e=zeros(k,l*bs);
e(end-l*bs+1:end,:)=eye(l*bs);
Ap(1:end+l*bs,end+1:end+l*bs)=...
    H(1:end-bs,1:end)*(K(1:end-bs,1:end)\e);
Ap(end-l*bs+1:end,1:end)=...
    H(end-(l+1)*bs+1:end-bs,1:end)/K(1:end-bs,1:end);
end

function np = newpole_adaptive_ADM(a,b,eigenvaluesA,eigenvaluesB, poles)
%Computes the newpole for rational Krylov method maximizing the determinant.

poles = poles(:);

eigenvaluesB = eigenvaluesB(:);

bs = length(eigenvaluesB) / length(poles);
poles = kron(ones(bs, 1), poles);

if isreal(eigenvaluesA)
    eHpoints=sort([a;b;eigenvaluesA]);
    maxvals=zeros(2,length(eHpoints)-1);
    for j=1:length(eHpoints)-1
        t=linspace(eHpoints(j),eHpoints(j+1),20);
        [maxvals(1,j),jx]= max(abs(ratfun(t, eigenvaluesB, poles)));
        maxvals(2,j)=t(jx);
    end
    [~,jx]=max(maxvals(1,:));
    np=maxvals(2,jx);
else
    x=[eigenvaluesA;a;b];
    k = convhull(real(x),imag(x));
    maxvals=zeros(2,length(k)-1);
    for i=1:length(k)-1
        t=linspace(x(k(i)),x(k(i+1)),20);
        [maxvals(1,i),jx] =max(abs(ratfun(t, eigenvaluesB, poles)));
        maxvals(2,i)=t(jx);
    end
    [~,jx]=max(maxvals(1,:));
    np=maxvals(2,jx);
end
end


function np = newpole_adaptive_sADM(a,b,eigenvaluesA, eigenvaluesB, poles)
%Computes the newpole for rational Krylov method maximizing the product
%of a selected set of eigenvalues.

poles = poles(:);

eigenvaluesB = eigenvaluesB(:);
bs = length(eigenvaluesB) / length(poles);

if isreal(eigenvaluesA)
    eHpoints=sort([a;b;eigenvaluesA]);
    maxvals=zeros(2,length(eHpoints)-1);
    for i=1:length(eHpoints)-1
        t=linspace(eHpoints(i),eHpoints(i+1),20);
        vals=zeros(1,length(t));
        for j=1:length(t)
            [~,I]=sort(abs(t(j)-eigenvaluesB), 'ascend');
            sorteig=eigenvaluesB(I);
            vals(j)=abs(prod( (t(j)-poles(2:end))./(t(j)-sorteig(bs+1:bs:end)) )...
                /((t(j)-sorteig(1))));
        end
        [maxvals(1,i),jx]= max(vals);
        maxvals(2,i)=t(jx);
    end
    [~,jx]=max(maxvals(1,:));
    np=maxvals(2,jx);

else
    x=[eigenvaluesA;a;b];
    k = convhull(real(x),imag(x));
    maxvals=zeros(2,length(k)-1);
    for i=1:length(k)-1
        t=linspace(x(k(i)),x(k(i+1)),20);
        vals=zeros(1,length(t));
        for j=1:length(t)
            [~,I]=sort(abs(t(j)-eigenvaluesB), 'ascend');
            sorteig=eigenvaluesB(I);
            vals(j)=abs(prod( (t(j)-poles(2:end))./(t(j)-sorteig(bs+1:bs:end)) )...
                /((t(j)-sorteig(1))));
        end
        [maxvals(1,i),jx] = max(vals);
        maxvals(2,i)=t(jx);
    end
    [~,jx]=max(maxvals(1,:));
    np=maxvals(2,jx);
end

end

function r=ratfun(x,eH,s)

r=zeros(size(x));

for j=1:length(x)
    r(j)=abs(prod( (x(j)-s(2:end))./(x(j)-eH(2:end)) )/((x(j)-eH(1))));
end

return

end
