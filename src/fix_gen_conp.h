/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Version Sep/22/2014
   Zhenxing Wang (KU)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(genconp, FixGenConp)

#else

#ifndef LMP_FIX_GEN_CONP_H
#define LMP_FIX_GEN_CONP_H

#include "fix.h"

namespace LAMMPS_NS {

class FixGenConp : public Fix {
 public:
  FixGenConp(class LAMMPS *, int, char **);
  ~FixGenConp();
  int setmask();
  void init();
  void setup(int);
  void pre_force(int);
  void force_cal(int);
  void a_cal();
  void a_read();
  void b_cal();
  void calc_struct_factors(const double &, const double &, const double &, double &, double &);
  void equation_solve();
  void update_charge();
  int electrode_check(int);
  void cg();
  void inv();
  void write_matrix(std::string);
  void write_vector(std::string);
  void coul_cal(int, double *, int *);

 private:
  int me, runstage;
  double Btime, Btime1, Btime2;
  double Ctime, Ctime1, Ctime2;
  double Ktime, Ktime1, Ktime2;
  double cgtime, cgtime1, cgtime2;
  FILE *outf, *outa, *a_matrix_fp;
  int a_matrix_f;
  int minimizer;
  double vL, vR;
  int molidL, molidR;
  int maxiter;
  double tolerance;

  double rms(int, double, bigint, double);
  double offdiag(const double &, const double &, const double &, const double &, const double &,
                 const double &, const double, const double, const double &);
  double b_component(const double &R_x, const double &R_y, const double &R_z, const double &Q_i,
                     const double &alpha, const double &eta, const double &k_max);
  double ion_sum(const double &R_x, const double &R_y, const double &R_z, double &alpha);
  void coeffs();

  double unitk[3];
  double *ug;
  double g_ewald, eta, gsqmx, volume, slab_volfactor;
  int *kxvecs, *kyvecs, *kzvecs;
  double ***cs, ***sn, **csk, **snk;
  int kmax, kmax3d, kmax_created, kcount;
  int kxmax, kymax, kzmax;
  double *sfacrl, *sfacrl_all, *sfacim, *sfacim_all;
  int everynum;
  int nr_electrode_atoms, nr_electrode_atoms_old, nr_electrode_atoms_all;
  double *eleallq;
  std::vector<double> aaa_all, bbb_all;
  int *id_to_electrode_id, *global_electrode_ids, *curr_id_to_electrode_id, *local_electrode_ids;
  std::vector<double> ks_x;
  std::vector<double> ks_y;
  std::vector<double> ks_z;
};

}    // namespace LAMMPS_NS

#endif
#endif
