function [Z,i,k,maxres] = restarted_shiftedFOM(A,s,b,tol,m,kmax)
% function [Z,i,k,maxres] = restarted_shiftedFOM(A,s,b,tol,m,kmax)
% Function the solves the shifted linear systems
% 
% (A+s_iI)x=b
% 
% by restarted FOM.
%
% INPUT
% A: nxn coefficient matrix
% s: vector of l shifts
% b: rhs
% tol: tolerance on the relative residual norm
% m: maximum number of iterations before restarting
% kmax: maximum number of restarting cycles
%
% OUTPUT
% Z: nxl matrix containing all the solutions column-wise
% i: total number of iterations in the last performed restarting cycle
% k: number of completed restarting cycles (overall number of iterations: m*(k-1)+i
% maxres: (m*(k-1)+i)-dimensional vector containing the convergence history, namely the 
%         maximum relative residual norm among all shifted linear systems computed along all iterations 
%
% Reference:
% Minimal Residual Rational Krylov Subspace Method for Sequences of Shifted Linear Systems
% Hussam Al Daas and Davide Palitta 
% ArXiv: 2507.00267
%
% Please, acknowledge our work by citing our manuscript whenever you use the software 
% provided here in your research.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
% INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
% PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
% HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
% TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
% OTHER DEALINGS IN THE SOFTWARE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


n = size(A, 1);
p=size(b,2);
l=length(s);

% preallocation
Z=zeros(n,l);
maxres=zeros(kmax*m,1);
    
% first basis vector
if p==1
    beta=norm(b);
    V=b/beta;
else
    [V(:, 1:p), beta] = qr(b, 0);
end
beta_vec=ones(l,1)*beta;


% restarting cycles
for k=1:kmax
    VV = V;

    H = zeros(m*p + 1, m * p);
    res_vec=zeros(l,1);

    for i=1:m

        W=A*V;

        % ortogonalize w w.r.t the previous basis vectors (re-orth modified gram-schmidt)
        for ll = 1:2
            k_min = max(1, i - m);
            for kk = k_min:i
                k1 = (kk - 1) * p + 1;
                k2 = kk * p;
                gamma = W' * VV(:, k1:k2);
                H(k1:k2, (i-1)*p+1:i*p) = H(k1:k2, (i-1)*p+1:i*p) + gamma;
                W = W - VV(:, k1:k2) * gamma;
            end
        end

        % orthonormalize the last basis vector
        if i <= m
            if p==1
                H(i*p+1:(i+1)*p, (i-1)*p+1:i*p)=norm(W);
                V=W/H(i*p+1:(i+1)*p, (i-1)*p+1:i*p);
            else
                [V, H(i*p+1:(i+1)*p, (i-1)*p+1:i*p)] = qr(W, 0);
            end
        end




        % solve the projected problem
        Y=zeros(i*p,l);
        HH=H(1:i*p, 1:i*p);
        e=zeros(i*p,1);
        e(1)=1;
        
        for j=1:l    
            Y(:,j) = (HH+s(j)*speye(i*p))\(e * beta_vec(j));
            res_vec(j)=H(i*p + 1:(i+1)*p, i*p) * abs(Y(i*p,j))/beta;%/abs(beta_vec(j));
        end
        conv_check=find(res_vec>tol);
           
        maxres((k-1)*m+i)=max(res_vec);

       if isempty(conv_check)
           Z=Z+VV(:,1:i*p)*Y;
           maxres=maxres(1:(k-1)*m+i);
           return
       end

        VV(:,i*p+1:(i+1)*p)=V;

    end

   % get ready to restart     
   beta_vec=-H(i*p + 1:(i+1)*p, i*p) * Y(end,:);
   % update solution before restarting
   Z=Z+VV(:,1:i*p)*Y;
end



