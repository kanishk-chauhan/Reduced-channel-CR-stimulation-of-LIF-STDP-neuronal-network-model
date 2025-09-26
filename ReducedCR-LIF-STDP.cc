/*                  
 *        File: ReducedCR-LIF-STDP.cc
 *      Author: Kanishk Chauhan
 *        Date: Jan 2025
 * Description: This code picks up the spontaneous steady-state data file and extracts all variables like membrane potentials,
 *              firing rates, posiitons of neurons, spike trains, etc. Procedure: 10 minutes of relaxation without stimulation, 
 *              followed by 1 minute of stimulation, and then by 1/2 an hour without stimulation. These durations can be changed.
 *              The code includes functions to produce various versions of Coordinated Reset stimulus, e.g., sequential, rapidly 
 *              varying sequences, sequential with jitters, rapidly varying sequences with jitters, and uncorrelated multichannel.
 */

# include <iostream>
# include <algorithm>
# include <iomanip>
# include <sstream>
# include <cmath>
# include <cstdlib>
# include <fstream> 
# include <vector>
# include <complex>
# include <random>

using namespace std;

// ----------------------TAKEN FROM NUMERICAL RECIPES FOR GENERATING RANDOM NUMBERS--------------------------------
struct Ran {
    typedef unsigned int long long Ullong; 
    typedef unsigned int Uint;
    Ullong u,v,w;
    Ran(Ullong j) : v(4101842887655102017LL), w(1) {
    u = j ^ v; int64();
    v = u; int64();
    w = v; int64();
}
inline Ullong int64() {
    u = u * 2862933555777941757LL + 7046029254386353087LL;
    v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
    w = 4294957665U*(w & 0xffffffff) + (w >> 32);
    Ullong x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
    return (x + v) ^ w;
}
inline double doub() { return 5.42101086242752217E-20 * int64(); } //Returns a double-type random number between 0 and 1
inline Uint int32() { return (Uint)int64(); } //Return 32-bit random integer.
};

struct Normaldev_BM : Ran {
    typedef double Doub;
    typedef unsigned int long long Ullong;
    Doub mu,sig;
    Doub storedval;
    Normaldev_BM(Doub mmu, Doub ssig, Ullong i) // constructor
    : Ran(i), mu(mmu), sig(ssig), storedval(0.) {}

    Doub dev() {
        Doub v1,v2,rsq,fac;
        if (storedval == 0.) {
            do {
                v1=2.0*doub()-1.0;
                v2=2.0*doub()-1.0;
                rsq=v1*v1+v2*v2;
            } while (rsq >= 1.0 || rsq == 0.0);
            fac=sqrt(-2.0*log(rsq)/rsq);
            storedval = v1*fac;
            return mu + sig*v2*fac;
        } else {
            fac = storedval;
            storedval = 0.;
            return mu + sig*fac;
        }
    }
};
// ------------------------------------------------------------------------------------------------------------------


// the function to generate Random Connectivity by Renyi-Erdos method
vector<vector<double> > Adjacency_Matrix(int N, double p);
/// the function to generate a the adjacency matrix with distance dependent connection probability  
vector<vector<double> > Dist_Dep_Adjacency_Matrix(int N, double p, vector<vector<double> > Zd);

// Random Reset Stimulus
vector<vector<double> > RRstimulus(Ran &random, double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset);
// Sequential CR
vector<vector<double> > SeqCR(Ran &random,double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset, int n_pulses);
// CR with Rapidly Varying Sequences
vector<vector<double> > RVS_CR(Ran &random,double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset, int n_pulses);
// Sequential CR with jitters in onset times
vector<vector<double> > jSeqCR(Ran &random,double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset, double jitter_strength, int n_pulses, int nmask);
// CR with Rapidly Varying Sequences and jitters in onset times
vector<vector<double> > jRVS_CR(Ran &random,double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset, double jitter_strength, int n_pulses, int nmask);

// the function to update the neurons' membrane potential
void InF(double dt, int itrain, double gleak[], double gsyn[], double gnoise[], double noise[], double V[], double Vth[], double Istim[], double inlinks[], double train_pre[], 
                double tdelay, double Vrest, double Vsyn, double Vth_rest, double Tsyn, double Tth, double C0, double k, double knoise, int N, 
                vector<vector<double> > &A, vector<vector<double> > &W, vector<vector<int> > &iLinks);

// the function to implement the STDP rule with TRACE AND TRAIN METHOD
void STDP1(int N, double dt, double ed, double ep, double train_pre[], double train_post[], double trace_pre[], double trace_post[], double inlinks[], vector<vector<double> > &W, vector<vector<int> > &iLinks);

// faster implementation of STDP
void fastSTDP(int N, double dt, double ed, double ep, double train_pre[], double train_post[], double trace_pre[], double trace_post[], double inlinks[], double outlinks[], vector<vector<double> > &W, vector<vector<int> > &iLinks, vector<vector<int> > &oLinks, vector<int> &pre_spikers, vector<int> &post_spikers);


const int m = 32;
const int N = m*m; // number of oscillators

int main(int argc, char* argv[]){

    double Tw,tp,tr,td,t,dt,dt_inv,msecs,tmax,pi,d_theta,d_theta_wrapped,ed,ep,epsilon,sum,a,f0,gamma,D,hours,w_sd,w0,W_mean,W_sd,dn,nl,n_con,n_con_init,knoise,fnoise,Tth,Tsyn,k,Vsyn,Vrest,Vth_rest,C0,Cstd,Vreset,Vspike,Tspike,Vth_spike,tdelay,Tslow;
    double seed, rnd, current_time,Avg_W,node_deg,order_par,rand_num,lamda,Pa,I,p,beta_min,beta_max,n_con_max,n_con_min,n_con_p,Fs,e,ki0,ki0N,ki0p,ki0Np,eta,tou_st,stim_sigma,b,H,sum_loc,ee,d0,lifetime_hole,P0,SI,Wmin,dd,L,dc,rtime0,fi_mean,fi_sd,spk;
    double gleak0, gleak_std, fT, fstd, odpr, sumsin, sumcos, hr, un, vn, dst, sigma_s, Ph, Pw, fi_mean0, Avg_W0, fi_err, Avg_Werr, max_err, stime, pstime, h_relx, jitter_strength;
    int nmask,nt,ntrain,itrain,n_noise_spikes,i_noise_spike,num_itrn,pos,pos_,pre,np,pruned,added,index,Na,a_init_pos,a_final_pos,part,rtime,stim_width,frac,osc_index,win_size,nspk,nlo,flg,sp_it,nrelx,nrelxNstim,n_pulses;
    double V[N], V0[N], Vth[N], C[N], gleak[N], theta[N], fi[N], cpl_term[N], theta_new[N], gnoise[N], Istim[N], Inoise[N], noise[N], gsyn[N], flag[N], trace_pre[N];
    double trace_post[N], noise_train_array[N], inlinks[N], outlinks[N], osc_indices[N], avgW[N], spikeTimes[N], spikeTimes0[N], train_pre[N], train_post[N], phi[N];
    double xi[N], yi[N];
    int counts[N], counts2[N];
    double c1,f1,f2,lambda_h,lambda_w,delta_f,nu,p0; // variables for structural plasticity
    double net_dim, stim_radius, stim_net_fraction, deltaV, Astim; // variables for stimulation
    vector<int> pre_spikers, post_spikers;

    // command line inputs
    Astim = atof(argv[1]); // stimulus intensity
    Fs = atof(argv[2]); // stimulus frequency
    nmask = atoi(argv[3]); // mask 'nmask' channels/electrodes, i.e., prevent this many channels from delivering stimulus

    char stimulus_variant[] = "jrvs"; // could be "seq", "rvs", "jseq", "jrvs", "rr" -- this variable is needed to create a correct file name for storing data
    string stim_variant(stimulus_variant); // -- this variable is required for using 'if' condition to choose the correct function to generate the stimulus
    n_pulses = 5;
    jitter_strength = 0.0;
    
    // the files to store all the data
    char filename1[500] = {};
    char filename2[500] = {};
    char filename3[500] = {};
    char filename4[500] = {};
    string Address = "/Users/chauhank/Documents/InF_2025/GitHub folder/";

    if (n_pulses > 1){
        snprintf(filename1,sizeof(filename1),"burst_stdp_%s_(Astim=%.2f,Fs=%.1f,mask=%d)largeNet.dat",stimulus_variant,Astim,Fs,nmask);
    }
    else{
        snprintf(filename1,sizeof(filename1),"pulsed_stdp_%s_(Astim=%.2f,Fs=%.1f,mask=%d)largeNet.dat",stimulus_variant,Astim,Fs,nmask);
    }
    
    string file1 = Address + filename1;
    ofstream data(file1);

    // time stuff
    h_relx = 5.0/60.0; // hours for relaxation with STDP-only before stimulation
    stime = 1; // stimulation time in minutes -  can be increased in steps of 1 minute, not less
    stime = stime/60.0; // stimulation time in hours
    pstime = 0.5; // post stimulation time in hours (with STDP-only)
    hours = h_relx + stime + pstime; // total number of hours to simulate
    msecs = 3600e3; // number of miliseconds in 1 hour
    dt = 0.1; // in ms
    dt_inv = 1/dt; // the height of a spike
    rtime = hours*msecs; // run time with plasticity (in ms)
    Tw = 6e3; 
    nt = Tw/dt; // number of time steps in one Tw long window
    nrelx = h_relx*msecs/Tw;
    nrelxNstim = (h_relx+stime)*msecs/Tw;

    // connectivity constraints
    fT = 4.5; // Target firing rate in Hz
    f0 = 3.0; // network-averaged firing rate in units of spike/time
    fstd = 0.3; // allowed standard deviaiton in the uncoupled neurons' firing rates
    d0 = 1.0/(N-1); // for stimulus spatial profile
    net_dim = 1.0; // in mm or any other suitable units of length
    
    // parameter values
    a = 1.4; // the asymmetry
    k = 8.0; // in mS/cm^2
    C0 = 3; // mean neuron membrane capacitance in units of micro F/cm^2
    gleak0 = (f0 - 0.9177303000633497)/125.6691632725962;  // these 'm' and 'c' for fr=m*gleak+c will change with noise intensity.
    gleak_std = fstd/125.6691632725962;
    Vrest = -38.0; // resting potential in mV
    Vreset = -67.0; // in mV
    Vsyn = 0.0; // in mV
    Vth_spike = 0.0; // in mV
    Vth_rest = -40.0; // in mV
    deltaV = Vth_spike - Vreset; // not used anywhere; RRstim function has delV, which is the same
    Vspike = 20.0; // in mV
    Tspike = 1.0; // in ms
    Tsyn = 1.0; // in ms
    Tth = 5.0; // in ms
    tdelay = 3.; // in ms
    Tslow = 30e3; // in ms
    knoise = 0.06; // in mS/cm^2
    fnoise = 20; // in Hz
    tr = 4.0; // scales the depression time constant wrt potentiation
    tp = 10.0; // in ms
    td = tr*tp;
    eta = 0.02; // weight change quantum
    ep = eta;
    ed = a*eta/tr; // As in Kromer and Tass's PRR 2020 Paper (Long-lasting desynchronization by decoupling stimulus)
    W_sd = 0.05; // standard deviation of weights
    win_size = 10; // 5 on the right and 5 on the left
    if(win_size%2==0) win_size+=1; // to keep the window size odd
    pi = 4*atan(1);
    max_err = 1e-3;

    // for random numbers
    seed = 100;
    Ran random(seed); //seed is given as a seed for random number generator struct. Now random.doub() would generate a uniform random number between 0 and 1
    Normaldev_BM normal_random(0,1,seed); // mean = 0, s.d. = 1, and seed = seed. Now normal_random.dev() would generate a normally distributed random number generated by B-M method

    //***  reading the staedy state adjacency mat (N^2 elements), weights (N^2 elements), ang.freq.(N elements), and phases (N elements)  ***
    vector<double> zeros_(N,0);
    vector<vector<double> > W(N,zeros_);
    vector<vector<double> > A(N,zeros_); // the Adjacency Matrix
    
    char inputfilename[500] = {};
    string filePath = "/Users/chauhank/Documents/InF_2025/GitHub folder/";
    snprintf(inputfilename,sizeof(inputfilename),"Steady_state_STDP(a=1.4,Wmean=0.9,k=8,beta=0.100)largeNet.dat");  
    string inputfile = filePath + inputfilename;
    ifstream SteadyStateFile(inputfile);
    if(SteadyStateFile.fail()){
        cout << "Couldn't Open the Steady State File" << endl;
        return 0;
    } 
    int NN = 2*N*N + 13*N;
    double* steadystate = new double[NN];
    int plc = 0;
    while(!SteadyStateFile.eof() && plc < NN){ SteadyStateFile >> steadystate[plc]; plc++; }
    SteadyStateFile.close();
    
    plc = 0;
    for(int ii = 0; ii < N; ii++) {xi[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {yi[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {gleak[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {V[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {Vth[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {fi[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {gsyn[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {gnoise[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {flag[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {train_pre[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {train_post[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {trace_pre[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) {trace_post[ii] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) for(int jj = 0; jj < N; jj++) {A[ii][jj] = steadystate[plc]; plc++;}
    for(int ii = 0; ii < N; ii++) for(int jj = 0; jj < N; jj++) {W[ii][jj] = steadystate[plc]; plc++;}

    //************************************************************************************************************************************

    // distance between neurons
    hr = net_dim/(m-1); // m divisions in 1.0mm
    dst = 0.5*net_dim;
    vector<double> zerodis(N,0);
    vector<vector<double> > dij(N,zerodis);
    vector<vector<double> > Zd(N,zerodis);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            dij[i][j] = sqrt(pow((xi[i]-xi[j]),2)+pow((yi[i]-yi[j]),2)); // xi[i] and yi[i] give the position of the i-th neuron
            Zd[i][j] = exp(-dij[i][j]/dst); // the distance-dependent factor that scales the probability of connecting two neurons at distance dij
        }
    }

    // Sites of stimulation
    const int Nc = 4; // number of clusters/sites
    double xsi[Nc] = {net_dim/4,3*net_dim/4,3*net_dim/4,net_dim/4}; // the x and y positions of the sites of stimulation
    double ysi[Nc] = {net_dim/4,net_dim/4,3*net_dim/4,3*net_dim/4};

    sigma_s = 0.15*net_dim; // corresponds to stim_radius=0.45mm when net_dim=1mm
    vector<vector<double> > dij_stim(Nc,zerodis);
    vector<vector<double> > Zd_stim(Nc,zerodis);
    for (int i = 0; i < Nc; i++){
        for (int j = 0; j < N; j++){
            dij_stim[i][j] = sqrt(pow((xsi[i]-xi[j]),2)+pow((ysi[i]-yi[j]),2)); // i-th row contains the distance of neurons from the i-th site
            Zd_stim[i][j] = exp(-dij_stim[i][j]*dij_stim[i][j]/(2*sigma_s*sigma_s)); // the spatial drop factor that scales the amount of stimulus the neurons at distance dij receive
        }
    }
    vector<vector<double> > stim;

    // the 2D 'spike train' array- rows= neurons, columns= corresponding spike trains
    ntrain = int(tdelay/dt); if(ntrain == 0) ntrain = 1; // number of points we need in the spike trains of all neurons
    vector<double> seero(ntrain,0);
    vector<vector<double> > train(N,seero);

    // the 2D 'inter-spike interval' array- rows= neurons, columns= corresponding inter-spike intervals
    vector<double> zeroisi(nt,0);
    vector<vector<double> > isi(N,zeroisi);

    vector<vector<double> > A0 = A; 

    // the 2D array to store the presynaptic partner information (neuron number) for all neurons
    vector<int> sero(N,0);
    vector<vector<int> > iLinks(N,sero);
    vector<vector<int> > oLinks(N,sero);

    // initializing some variables arrays we might use later 
    for(int i = 0; i < N; i++){
        spikeTimes[i] = 0;
        counts2[i] = 0;
    }

    // for the 2D 'noise spike time intervals' array- rows= neurons, columns= corresponding noise inter-spike intervals
    n_noise_spikes = int(10*Tw/1000*fnoise); //cout << n_noise_spikes << endl;
    
    // calculating the INITIAL mean of the synaptic weights and node degree
    sum = 0; n_con = 0;
    for(int ii = 0; ii < N; ii++){
        nl = 0; pos = 0; pos_ = 0; sum_loc = 0; nlo = 0;
        for(int jj = 0; jj < N; jj++){
            if (A[ii][jj] == 1){
                sum_loc += W[ii][jj];
                iLinks[ii][pos] = jj;
                nl++;
                pos++;
            }
            if (A[jj][ii] == 1){
                oLinks[ii][pos_] = jj;
                nlo++;
                pos_++;
            }
        }
        outlinks[ii] = nlo;
        inlinks[ii] = nl;
        n_con += nl;
        sum += sum_loc;
        avgW[ii] = sum_loc/nl;
    }
    n_con_init = n_con; // this should not be updated after this step
    Avg_W = sum/n_con;
    node_deg = n_con/N/(N-1); 
    pruned = added = 0;

    //********************************************* STIMULATION and relaxation afterwards ************************************************//
    num_itrn = rtime/Tw;  
    for (int i = 0; i < num_itrn; i++){

        if (i >= nrelx && i < nrelxNstim){
            // the following functions return a stimulus vector of shape(Nsites,nt)
            if (stim_variant == "rr") stim = RRstimulus(random,Tw,Fs,dt,Nc,Astim,C0,Vth_spike,Vreset);
            if (stim_variant == "seq") stim = SeqCR(random,Tw,Fs,dt,Nc,Astim,C0,Vth_spike,Vreset,n_pulses);
            if (stim_variant == "jseq") stim = jSeqCR(random,Tw,Fs,dt,Nc,Astim,C0,Vth_spike,Vreset,jitter_strength,n_pulses,nmask);
            if (stim_variant == "rvs") stim = RVS_CR(random,Tw,Fs,dt,Nc,Astim,C0,Vth_spike,Vreset,n_pulses);
            if (stim_variant == "jrvs") stim = jRVS_CR(random,Tw,Fs,dt,Nc,Astim,C0,Vth_spike,Vreset,jitter_strength,n_pulses,nmask);
        }
        odpr = 0;
        for(int ii = 0; ii < N; ii++) counts[ii] = 0;
        
        // the 2D 'noise train' array- rows= neurons, columns= corresponding noise spike trains
        vector<double> zeronoise(nt,0);
        vector<vector<double> > noise_train(N,zeronoise);
        // preparing the noise spike train using the exponentially distributed random inter-spike intervals
        for (int ii = 0; ii < N; ii++){ // for each neuron
            i_noise_spike = 0; // index of spike in the noise spike train
            for (int jj = 0; jj < n_noise_spikes; jj++){ 
                rnd =  -log(random.doub())/fnoise; 
                i_noise_spike += int(rnd*1000/dt);
                if (i_noise_spike >= nt) break;
                noise_train[ii][i_noise_spike] = dt_inv;
            }
        }
        
        // for each time step of the iteration
        for (int j = 0; j < nt; j++){
            current_time = i*Tw + j*dt;
            itrain = int(j%ntrain); // the index for time 'tdelay' before the current time 
            pre_spikers.clear();
            post_spikers.clear();

            // presynaptic train and trace
            for (int ii = 0; ii < N; ii++){ // for each neuron
                train_pre[ii] = train[ii][itrain]; // delayed spike train
                trace_pre[ii] = trace_pre[ii] + dt*(-trace_pre[ii]/tp + train_pre[ii]); // updating the presynaptic trace with the spike train 'tdelay' before the current time
            }

            // needed for InF update
            for (int ii = 0; ii < N; ii++){
                V0[ii] = V[ii];
                noise[ii] = noise_train[ii][j];
                Istim[ii] = 0;
            }

            // upto 'h_relx' no stim, then stim for 'stime', followed by no stim.
            if (i >= nrelx && i < nrelxNstim){
                for(int ii = 0; ii < N; ii++){ // for each neurons
                    for(int jj = 0; jj < Nc; jj++){ // add the stimulus from all sites (note that the stimulus itself can be zero, positive, or negative at any given time)
                        Istim[ii] += stim[jj][j]*Zd_stim[jj][ii];
                    }
                }
            }
            // integrating the Integrate and Fire model equations
            InF(dt,itrain,gleak,gsyn,gnoise,noise,V,Vth,Istim,inlinks,train_pre,tdelay,Vrest,Vsyn,Vth_rest,Tsyn,Tth,C0,k,knoise,N,A,W,iLinks);
            
            // discriminating the spikes
            for (int ii = 0; ii < N; ii++){ // for each neuron
                train[ii][itrain] = 0;  // the train gets updated in this loop. Upto the step before this, the train is from 'tdelay' before the current time
                if (train_pre[ii] != 0) pre_spikers.push_back(ii);
                if (V[ii] >= Vth[ii] && flag[ii] == 1){
                    spikeTimes0[ii] = spikeTimes[ii];
                    spikeTimes[ii] = current_time;
                    post_spikers.push_back(ii);
                    isi[ii][counts[ii]] = spikeTimes[ii] - spikeTimes0[ii];
                    counts[ii] += 1; counts2[ii] += 1;
                    train[ii][itrain] = dt_inv;
                    Vth[ii] = Vth_spike;
                    V[ii] = Vspike;
                    flag[ii] = 0;
                }
                if (flag[ii] == 0){
                    if (j < (spikeTimes[ii] - i*Tw + Tspike)/dt){ // generating a rectangular spike of width Tspike
                        V[ii] = Vspike; 
                        Vth[ii] = Vth_spike;
                    }
                    else{
                        V[ii] = Vreset;
                        flag[ii] = 1;
                    }
                }
            }
            // postsynaptic train and trace
            for (int ii = 0; ii < N; ii++){ // for each neuron
                train_post[ii] = train[ii][itrain]; // current (non-delayed) spike train for postsynaptic neuron
                trace_post[ii] = trace_post[ii] + dt*(-trace_post[ii]/td + train_post[ii]); // updating the postsynaptic trace with the spike train at the current time
            }

            // the firing rate
            for (int ii = 0; ii < N; ii++){
                fi[ii] += dt/Tslow * (-fi[ii] + train_post[ii]*1e3);
            }

            // order parameter
            if (i > 0){
                sumsin = 0; sumcos = 0;
                for (int ii = 0; ii < N; ii++){
                    phi[ii] = 2*pi*((current_time - spikeTimes0[ii])/(spikeTimes[ii] - spikeTimes0[ii]) + counts2[ii]-1);
                    sumcos += cos(phi[ii]); sumsin += sin(phi[ii]);
                }
                odpr += sqrt(pow((sumcos/N),2) + pow((sumsin/N),2));
            }
            
            // if (!pre_spikers.empty() || !post_spikers.empty()) STDP1(N,dt,ed,ep,train_pre,train_post,trace_pre,trace_post,inlinks,W,iLinks);// updating the weights
            if (!pre_spikers.empty() || !post_spikers.empty()) fastSTDP(N,dt,ed,ep,train_pre,train_post,trace_pre,trace_post,inlinks,outlinks,W,iLinks,oLinks,pre_spikers,post_spikers); 
        }
        odpr = odpr/nt;  // time-averaging the order paramter

        fi_mean = 0;
        for (int ii = 0; ii < N; ii++) fi_mean += fi[ii]/N;  // mean firing rate
        
        // calculating the mean of the synaptic weights and node degree
        sum = 0; n_con = 0;
        for(int ii = 0; ii < N; ii++){
            nl = inlinks[ii]; sum_loc = 0;
            for(int jj = 0; jj < nl; jj++){
                pre = iLinks[ii][jj];
                sum_loc += W[ii][pre];
            }
            n_con += nl;
            sum += sum_loc;
            avgW[ii] = sum_loc/nl;
        }
        Avg_W = sum/n_con;
        node_deg = n_con/N/(N-1);

        data << left << setw(15) << odpr << left << setw(15) << fi_mean << left << setw(15) << Avg_W << left << setw(15) << node_deg << endl;      
        cout << odpr << "  " << fi_mean << "   " << Avg_W << "   " << node_deg << "   " << endl;  
    }

    data.close();
    return 0;
}

void InF(double dt, int itrain, double gleak[], double gsyn[], double gnoise[], double noise[], double V[], double Vth[], double Istim[], double inlinks[], double train_pre[], 
                double tdelay, double Vrest, double Vsyn, double Vth_rest, double Tsyn, double Tth, double C0, double k, double knoise, int N, 
                vector<vector<double> > &A, vector<vector<double> > &W, vector<vector<int> > &iLinks){
    
    double cpl_term;
    int nl, pre;
    for(int ii = 0; ii < N; ii++){ // updating the membrane potential and the threshold of each neuron
        V[ii] = V[ii] + dt/C0 * (gleak[ii]*(Vrest - V[ii]) + (gsyn[ii] + gnoise[ii])*(Vsyn - V[ii]) + Istim[ii]);
        Vth[ii] = Vth[ii] - dt/Tth * (Vth[ii] - Vth_rest);
    }
    for(int ii = 0; ii < N; ii++){ // updating the 'gsyn' of each neuron
        cpl_term = 0;
        nl = inlinks[ii];
        for (int jj = 0; jj < nl; jj++){ // evaluating the coupling term
            pre = iLinks[ii][jj];
            cpl_term += W[ii][pre] * train_pre[pre]; // train_pre[i] is either 0 or 1/dt
        }
        gsyn[ii] = gsyn[ii] + dt/Tsyn * (-gsyn[ii] + k*Tsyn*cpl_term/N);
    }
    for(int ii = 0; ii < N; ii++){ // updating the 'gnoise' of each neuron
        gnoise[ii] = gnoise[ii] + dt/Tsyn * (-gnoise[ii] + knoise*Tsyn*noise[ii]);
    }
     
}

void STDP1(int N, double dt, double ed, double ep, double train_pre[], double train_post[], double trace_pre[], double trace_post[], double inlinks[], vector<vector<double> > &W, vector<vector<int> > &iLinks){
    int nl,pre;
    for (int ii= 0; ii< N; ii++){ // for each postsynaptic neuron
        nl = inlinks[ii];
        for (int jj = 0; jj < nl; jj++){
            pre = iLinks[ii][jj];
            if (train_pre[pre] != 0 && train_post[ii] != 0) continue; // if both partners spike together, no weight change
            W[ii][pre] = W[ii][pre] + dt*(-ed*(trace_post[ii]*train_pre[pre]) + ep*(trace_pre[pre]*train_post[ii]));
            if (W[ii][pre] > 1) W[ii][pre] = 1;
            if (W[ii][pre] < 0) W[ii][pre] = 0;
        } 
    }
}

void fastSTDP(int N, double dt, double ed, double ep, double train_pre[], double train_post[], double trace_pre[], double trace_post[], double inlinks[], double outlinks[], vector<vector<double> > &W, vector<vector<int> > &iLinks, vector<vector<int> > &oLinks, vector<int> &pre_spikers, vector<int> &post_spikers){
    int nl,pre,post;
    double w_;
    for (int post:post_spikers){ // for each postsynaptic neuron, weight of contacts from its presynaptic partners increases whenever it spikes
        nl = inlinks[post];
        for (int jj = 0; jj < nl; jj++){
            pre = iLinks[post][jj];
            if (find(pre_spikers.begin(), pre_spikers.end(), pre) != pre_spikers.end()) continue;  // if both partners spike together, no weight change
            W[post][pre] = W[post][pre] + dt * ep * trace_pre[pre] * train_post[post];
        }
    }
    for (int pre:pre_spikers){ // for each presynaptic neuron, weight of contacts to its postsynaptic partners decreases whenever it spikes
        nl = outlinks[pre];
        for (int jj = 0; jj < nl; jj++){
            post = oLinks[pre][jj];
            if (find(post_spikers.begin(), post_spikers.end(), post) != post_spikers.end()) continue;  // if both partners spike together, no weight change
            W[post][pre] = W[post][pre] - dt * ed * trace_post[post] * train_pre[pre];
        }
    }
    for (int post:post_spikers){ // for each postsynaptic neuron, weight of contacts from its presynaptic partners increases whenever it spikes
        nl = inlinks[post];
        for (int jj = 0; jj < nl; jj++){
            pre = iLinks[post][jj];
            w_ = W[post][pre];
            if (w_ > 1) w_ = 1;
            if (w_ < 0) w_ = 0;
            W[post][pre] = w_;
        }
    }
    for (int pre:pre_spikers){ // for each presynaptic neuron, weight of contacts to its postsynaptic partners decreases whenever it spikes
        nl = outlinks[pre];
        for (int jj = 0; jj < nl; jj++){
            post = oLinks[pre][jj];
            w_ = W[post][pre];
            if (w_ > 1) w_ = 1;
            if (w_ < 0) w_ = 0;
            W[post][pre] = w_;
        }
    }
}

vector<vector<double> > RRstimulus(Ran &random,double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset){
    int nt, nte, nt_, nti, nt_lambda, i_pulse, n_pulses;
    double delV, t_e, t_, t_i, t_lambda, Fstim, Ae, Ai, t_s, Tstim, rnd;
    cout << "-------------- You are using RR stimulus --------------" << endl;
    delV = Vth_spike - Vreset;
    t_e = 0.5; // width of positive stimulus in ms
    t_ = 0.2;// width of gap between positive and negative peaks (in ms)
    t_i = 1.5; // width of negative stimulus in ms
    t_lambda = t_e + t_ + t_i; // total duration of a charge-balanced pulse that comprises of the pos and neg peaks (in ms)
    Fstim = 1.0/(1.0/Fs - 1.0/130); // because 'time' is in ms - This will be the effective average freq of stimulus
    Ae = Astim * delV * C0 / t_e; // peak amplitude of the positive (excitatory) stimulus 
    Ai = -Astim * delV * C0 / t_i ;// peak amplitude of the negative (inhibitory) stimulus 
    
    nt = window_size/dt;
    nte = t_e/dt; // number of time steps in one positive peak
    nt_ = t_/dt; // number of time steps between pos and neg
    nti = t_i/dt; // number of time steps in one negative peak
    nt_lambda = nte + nt_ + nti; // int(t_lambda/dt) # number of time steps in one charge-balanced pulse
    
    n_pulses = 10*window_size/1000*Fstim; 
    vector<double> zeros(nt,0);
    vector<vector<double> > Stimulus(Nsites,zeros);

    // preparing the stimulus trains using the exponentially distributed random inter-spike intervals
    for (int ii = 0; ii < Nsites; ii++){ // for each site
        i_pulse = 0; // index of pulse in the stimulus train (it should be set such that kk begins with at least 0, not negative)
        while (i_pulse < nt-nt_lambda){ 
            rnd = -log(random.doub())/Fstim; // exp_rnd(rnd_generator);
            if (rnd < 1.0/130) continue; // int(rnd*1e3/dt) < 1e3/130.0/dt
            i_pulse += int(rnd*1e3/dt);
            if (i_pulse >= nt-nt_lambda-1) break;
            for (int kk = i_pulse; kk < i_pulse+nte; kk++) Stimulus[ii][kk] = Ae; // the positive peak
            for (int kk = i_pulse+nte+nt_; kk < i_pulse+nt_lambda; kk++) Stimulus[ii][kk] = Ai; // the negative peak
        }
    }
        
    return Stimulus;
}

// Sequential CR
vector<vector<double> > SeqCR(Ran &random,double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset, int n_pulses){
    int nt, nte, nt_, nti, nt_lambda, i_pulse, nt_cycle, n_cycles, i_cycle;
    double delV, t_e, t_, t_i, t_lambda, Fstim, Ae, Ai, t_s, Tstim, rnd, Tcycle;

    cout << "-------------- You are using Sequential CR stimulus --------------" << endl;
    delV = Vth_spike - Vreset;
    t_e = 0.5; // width of positive stimulus in ms
    t_ = 0.2;// width of gap between positive and negative peaks (in ms)
    t_i = 1.5; // width of negative stimulus in ms
    t_lambda = t_e + t_ + t_i; // total duration of a charge-balanced pulse that comprises of the pos and neg peaks (in ms)
    Tcycle = 1e3/Fs; // total duration of a stimulus cycle
    Ae = Astim * delV * C0 / t_e; // peak amplitude of the positive (excitatory) stimulus 
    Ai = -Astim * delV * C0 / t_i ;// peak amplitude of the negative (inhibitory) stimulus 
    
    nt = window_size/dt;
    nte = ceil(t_e/dt); // number of time steps in one positive peak
    nt_ = ceil(t_/dt); // number of time steps between pos and neg
    nti = ceil(t_i/dt); // number of time steps in one negative peak
    nt_lambda = nte + nt_ + nti; // int(t_lambda/dt) # number of time steps in one charge-balanced pulse
    nt_cycle = Tcycle/dt; // # number of time steps in one stimulus cycle
    
    vector<double> zeros(nt,0);
    vector<vector<double> > Stimulus(Nsites,zeros);

    n_cycles = Fs*window_size/1e3; // because window_size is in ms
    i_cycle = 0;
    while (i_cycle < n_cycles-1){
        for (int ii = 0; ii < Nsites; ii++){
            i_pulse = i_cycle*nt_cycle + ii*nt_cycle/Nsites;  // pulse/burst onset
            for (int jj = 0; jj < n_pulses; jj++){ 
                for (int kk = i_pulse; kk < i_pulse+nte; kk++) Stimulus[ii][kk] = Ae; // the positive peak
                for (int kk = i_pulse+nte+nt_; kk < i_pulse+nt_lambda; kk++) Stimulus[ii][kk] = Ai; // the negative peak
                i_pulse += ceil(8.3/dt);
            }
        }
        i_cycle += 1;
    }
    return Stimulus;
}

// CR with Rapidly Varying Sequences
vector<vector<double> > RVS_CR(Ran &random,double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset, int n_pulses){
    int nt, nte, nt_, nti, nt_lambda, i_pulse, nt_cycle, n_cycles, i_cycle;
    double delV, t_e, t_, t_i, t_lambda, Fstim, Ae, Ai, t_s, Tstim, rnd, Tcycle;

    cout << "-------------- You are using RVS CR stimulus --------------" << endl;
    delV = Vth_spike - Vreset;
    t_e = 0.5; // width of positive stimulus in ms
    t_ = 0.2;// width of gap between positive and negative peaks (in ms)
    t_i = 1.5; // width of negative stimulus in ms
    t_lambda = t_e + t_ + t_i; // total duration of a charge-balanced pulse that comprises of the pos and neg peaks (in ms)
    Tcycle = 1e3/Fs; // total duration of a stimulus cycle
    Ae = Astim * delV * C0 / t_e; // peak amplitude of the positive (excitatory) stimulus 
    Ai = -Astim * delV * C0 / t_i ;// peak amplitude of the negative (inhibitory) stimulus 
    
    nt = window_size/dt;
    nte = ceil(t_e/dt); // number of time steps in one positive peak
    nt_ = ceil(t_/dt); // number of time steps between pos and neg
    nti = ceil(t_i/dt); // number of time steps in one negative peak
    nt_lambda = nte + nt_ + nti; // int(t_lambda/dt) # number of time steps in one charge-balanced pulse
    nt_cycle = Tcycle/dt; // # number of time steps in one stimulus cycle
    
    vector<double> zeros(nt,0);
    vector<vector<double> > Stimulus(Nsites,zeros);

    n_cycles = Fs*window_size/1e3; // because window_size is in ms
    i_cycle = 0;
    vector<int> pulse_order(Nsites,0);
    for (int ii = 0; ii < Nsites; ii++) pulse_order[ii] = ii;

    while (i_cycle < n_cycles-1){
        shuffle(pulse_order.begin(),pulse_order.end(),default_random_engine(0));
        for (int ii = 0; ii < Nsites; ii++){
            i_pulse = i_cycle*nt_cycle + pulse_order[ii]*nt_cycle/Nsites;
            for (int jj = 0; jj < n_pulses; jj++){ 
                for (int kk = i_pulse; kk < i_pulse+nte; kk++) Stimulus[ii][kk] = Ae; // the positive peak
                for (int kk = i_pulse+nte+nt_; kk < i_pulse+nt_lambda; kk++) Stimulus[ii][kk] = Ai; // the negative peak
                i_pulse += ceil(8.3/dt);
            }
        }
        i_cycle += 1;
    }
    return Stimulus;
}

// Sequential CR with random jitters
vector<vector<double> > jSeqCR(Ran &random,double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset, double jitter_strength, int n_pulses, int nmask){
    int nt, nte, nt_, nti, nt_lambda, i_pulse, nt_cycle, n_cycles, i_cycle, Nsites_, ii_;
    double delV, t_e, t_, t_i, t_lambda, Fstim, Ae, Ai, t_s, Tstim, rnd, Tcycle;

    cout << "-------------- You are using Sequential CR stimulus with jitters --------------" << endl;
    Nsites_ = Nsites - nmask;
    delV = Vth_spike - Vreset;
    t_e = 0.5; // width of positive stimulus in ms
    t_ = 0.2;// width of gap between positive and negative peaks (in ms)
    t_i = 1.5; // width of negative stimulus in ms
    t_lambda = t_e + t_ + t_i; // total duration of a charge-balanced pulse that comprises of the pos and neg peaks (in ms)
    Tcycle = 1e3/Fs; // total duration of a stimulus cycle (ms)
    Ae = Astim * delV * C0 / t_e; // peak amplitude of the positive (excitatory) stimulus 
    Ai = -Astim * delV * C0 / t_i ;// peak amplitude of the negative (inhibitory) stimulus 
    
    nt = window_size/dt;
    nte = ceil(t_e/dt); // number of time steps in one positive peak
    nt_ = ceil(t_/dt); // number of time steps between pos and neg
    nti = ceil(t_i/dt); // number of time steps in one negative peak
    nt_lambda = nte + nt_ + nti; // int(t_lambda/dt) # number of time steps in one charge-balanced pulse
    nt_cycle = Tcycle/dt; // # number of time steps in one stimulus cycle

    vector<double> zeros(nt,0);
    vector<vector<double> > Stimulus(Nsites,zeros);

    n_cycles = Fs*window_size/1e3; // because window_size is in ms
    i_cycle = 0;
    vector<double> mask_it(Nsites,0);
    for (int ii = 0; ii < Nsites; ii++) mask_it[ii] = ii; // mask_it element = 0 would mask the corresponding electrode/channel (i.e., make its current zero)

    while (i_cycle < n_cycles-1){
        shuffle(mask_it.begin(), mask_it.end(),default_random_engine(0)); // to randomly mask electrodes/channels
        i_pulse = i_cycle * nt_cycle + nt_cycle/Nsites_/2 * jitter_strength * random.doub(); // cycle onset
        for (int ii = 0; ii < Nsites; ii++){
            if (find(mask_it.begin(), mask_it.begin()+nmask, ii) != mask_it.begin()+nmask) continue;
            for (int jj = 0; jj < n_pulses; jj++){ 
                for (int kk = i_pulse; kk < i_pulse+nte; kk++) Stimulus[ii][kk] = Ae; // the positive peak
                for (int kk = i_pulse+nte+nt_; kk < i_pulse+nt_lambda; kk++) Stimulus[ii][kk] = Ai; // the negative peak
                i_pulse += ceil(8.3/dt);
            }
            i_pulse -= n_pulses * ceil(8.3/dt);
            i_pulse += nt_cycle/Nsites_;
        }
        i_cycle += 1;
    }
    return Stimulus;
}

// Rapidly varying sequences CR with random jitters
vector<vector<double> > jRVS_CR(Ran &random,double window_size, double Fs, double dt, int Nsites, double Astim, double C0, double Vth_spike, double Vreset, double jitter_strength, int n_pulses, int nmask){
    int nt, nte, nt_, nti, nt_lambda, i_pulse, nt_cycle, n_cycles, i_cycle, Nsites_, ii_;
    double delV, t_e, t_, t_i, t_lambda, Fstim, Ae, Ai, t_s, Tstim, rnd, Tcycle;

    cout << "-------------- You are using RVS CR stimulus with jitters --------------" << endl;
    Nsites_ = Nsites - nmask;
    delV = Vth_spike - Vreset;
    t_e = 0.5; // width of positive stimulus in ms
    t_ = 0.2;// width of gap between positive and negative peaks (in ms)
    t_i = 1.5; // width of negative stimulus in ms
    t_lambda = t_e + t_ + t_i; // total duration of a charge-balanced pulse that comprises of the pos and neg peaks (in ms)
    Tcycle = 1e3/Fs; // total duration of a stimulus cycle
    Ae = Astim * delV * C0 / t_e; // peak amplitude of the positive (excitatory) stimulus 
    Ai = -Astim * delV * C0 / t_i ;// peak amplitude of the negative (inhibitory) stimulus 
    
    nt = window_size/dt;
    nte = ceil(t_e/dt); // number of time steps in one positive peak
    nt_ = ceil(t_/dt); // number of time steps between pos and neg
    nti = ceil(t_i/dt); // number of time steps in one negative peak
    nt_lambda = nte + nt_ + nti; // int(t_lambda/dt) # number of time steps in one charge-balanced pulse
    nt_cycle = Tcycle/dt; // # number of time steps in one stimulus cycle
    
    vector<double> zeros(nt,0);
    vector<vector<double> > Stimulus(Nsites,zeros);

    n_cycles = Fs*window_size/1e3; // because window_size is in ms
    i_cycle = 0;
    vector<int> pulse_order(Nsites,0);
    vector<int> mask_it(Nsites,0);
    for (int ii = 0; ii < Nsites; ii++) pulse_order[ii] = ii;

    while (i_cycle < n_cycles-1){
        shuffle(pulse_order.begin(),pulse_order.end(),default_random_engine(0)); // for both channel/electrode masking and randomization of order of stimulation
        i_pulse = i_cycle * nt_cycle + nt_cycle/Nsites_/2 * jitter_strength * random.doub(); // cycle onset
        for (int ii = nmask; ii < Nsites; ii++){
            ii_ = pulse_order[ii];
            for (int jj = 0; jj < n_pulses; jj++){ 
                for (int kk = i_pulse; kk < i_pulse+nte; kk++) Stimulus[ii_][kk] = Ae; // the positive peak
                for (int kk = i_pulse+nte+nt_; kk < i_pulse+nt_lambda; kk++) Stimulus[ii_][kk] = Ai; // the negative peak
                i_pulse += ceil(8.3/dt);
            }
            i_pulse -= n_pulses * ceil(8.3/dt);
            i_pulse += nt_cycle/Nsites_;
        }
        i_cycle += 1;
    }
    return Stimulus;
}