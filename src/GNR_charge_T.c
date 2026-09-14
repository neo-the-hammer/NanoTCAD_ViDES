// ======================================================================
//  Copyright (c) 2004-2010, G. Fiori, G. Iannaccone  University of Pisa
//  
//  This file is released under the BSD license.
//  See the file "license.txt" for information on usage and
//  redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
// ====================================================================== 

//
// This subroutine computes the charge within the armchair nanoribbon 
// (unrolled zig-zag (n,0) Carbon Nanotubes as well as the
// transmission coefficient, within the real space approach.
//
// N.B.: Atoms are ordered first within the unrolled ring and then along the 
// GNRde r axis. E.G.: if i runs from 0 to n-1 (where n is the number of atoms
// along the unrolled ring) and j from 0 to Nc-1 
// (where Nc is the number of rings)
// the ix-th atom with index (i,j) has index ix=i+j*n
//
// INPUT:
//
// nanoribbon.n: number of atoms along unrolled zig-zag CNT ring
// nanoribbon.Nc: number of unrolled zig-zag CNT rings
// nanoribbon.Eupper [eV]: highest considered energy
// nanoribbon.Elower [eV]: lowest considered energy
// nanoribbon.dE [eV]: energy step
// nanoribbon.thop [eV]: hopping parameter
// nanoribbon.eta [eV]: broadening parameter
// nanoribbon.Temp [K]: GNR temperature
// nanoribbon.mu1 [eV]: Fermi level of the left reservoir
// nanoribbon.mu2 [eV]: Fermi level of the right reservoir
// nanoribbon.contact [char]: specifies if Schottky or doped reservoirs 
//                          have to be considered
// nanoribbon.Phi (vector n*Nc) [eV]: electrostatic potential in correspondence 
//                                  of each C atom ordered as above.
//
// OUTPUT:
//
// nanoribbon.charge (vector n*Nc): charge in correspondence of each atoms 
//                                (electrons-holes), i.e. if positive,
//                                then electrons, if negative then holes.
//                                Atoms are ordered as above.
// nanoribbon.E (vector) [eV]: computed energy levels stored in a vector
// nanoribbon.T (vector): nanoribbon.T[i] is the transmission coefficient 
//                      computed at nanoribbon.E[i].
//
//

#include "GNR_charge_T.h"
#include "vides_rgf_batch.h"
static PyObject* py_GNR_charge_T(PyObject* self, PyObject* args)
{
  int i,j,k,ix,ie,Nm,icse,NE,n,Nc,nx,rank;
  complex ***DIAG,***UPDIAG,***LOWDIAG;
  complex zero,**SIGMAS,**SIGMAD;
  double E,dE,Emin,Emax,*tempo,eta,
    **A1,**A2,**A1old,**A2old,*ncarcnt,T,*passos,*passod,arg1,arg2,
    sum,Eupper,Elower,*Phi,thop,*EE,*TE,mu1,mu2,vt,Egap,Temperature;
  PyObject *obj,*temp_obj;
  PyArrayObject *Phi_array;
  char *GNRboundary;

  setvbuf(stdout,(char *) NULL,_IONBF,1);

  // I read the data from python class
  import_array();
  if (!PyArg_ParseTuple(args,"O",&obj))
    {
      printf("NOT A CLASS! \n");
      exit(0);
    }
  temp_obj=PyObject_GetAttrString(obj,"rank");
  rank=(int)PyInt_AsLong(temp_obj);
  if (!rank) printf("*********************************\n");
  if (!rank) printf("*** Computing NEGF in GNR *******\n"); 
  if (!rank) printf("*********************************\n\n");
  temp_obj=PyObject_GetAttrString(obj,"n");
  n=(int)PyInt_AsLong(temp_obj);
  if (!rank) printf("n of the nanoribbon = %d \n",n);
  temp_obj=PyObject_GetAttrString(obj,"Nc");
  Nc=(int)PyInt_AsLong(temp_obj);
  if (!rank) printf("Number of rings of the nanoribbon = %d \n",Nc);
  temp_obj=PyObject_GetAttrString(obj,"Eupper");
  Eupper=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Upper Energy defined by the user = %lg \n",Eupper);
  temp_obj=PyObject_GetAttrString(obj,"Elower");
  Elower=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Lower Energy defined by the user = %lg \n",Elower);
  temp_obj=PyObject_GetAttrString(obj,"dE");
  dE=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Delta Energy = %lg \n",dE);
  temp_obj=PyObject_GetAttrString(obj,"thop");
  thop=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Hopping parameter = %lg \n",thop);
  temp_obj=PyObject_GetAttrString(obj,"eta");
  eta=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Eta = %lg \n",eta);
  temp_obj=PyObject_GetAttrString(obj,"Temp");
  Temperature=(double)PyFloat_AsDouble(temp_obj);
  vt=kboltz*Temperature/q;
  if (!rank) printf("Temperature = %lg \n",Temperature);
  temp_obj=PyObject_GetAttrString(obj,"mu1");
  mu1=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Electrochemical potential of the source = %lg \n",mu1);
  temp_obj=PyObject_GetAttrString(obj,"mu2");
  mu2=(double)PyFloat_AsDouble(temp_obj);
  if (!rank) printf("Electrochemical potential of the drain = %lg \n",mu2);
  temp_obj=PyObject_GetAttrString(obj,"contact");
  #if PY_MAJOR_VERSION >=3
  GNRboundary=PyBytes_AsString(PyUnicode_AsEncodedString(temp_obj,"utf-8","Error"));
  #else
  GNRboundary=PyString_AsString(temp_obj);
  #endif
  if (strcasecmp(GNRboundary,"doped")==0)
    if (!rank) printf("GNR contacts : doped reservoirs \n");
  else if (strcasecmp(GNRboundary,"schottky")==0)
    if (!rank) printf("GNR contacts : Schottky contacts \n");
  else
    {
      if (!rank) printf("Error in the definition of the GNR contact \n");
      if (!rank) printf("It can be either doped or schottky \n");
      exit(0);
    }

  Phi=dvector(0,n*Nc-1);
  for (i=0;i<n*Nc-1;i++)
    Phi[i]=0;
  Phi_array=(PyArrayObject*)PyObject_GetAttrString(obj,"Phi");
  nx=Phi_array->dimensions[0];
  for (i=0;i<nx;i++)
    {
      tempo=(double *)(Phi_array->data+i*Phi_array->strides[0]);
      Phi[i]=*tempo;
    }
  
  Egap=gappo(n,fabs(thop));
  if (!rank) printf("Egap of the nanoribbon = %lg \n",Egap);
  if (!rank) printf("****************************** \n");

  // Variable initialization and creation
  ie=0;
  zero.r=0;
  zero.i=0;
  

  Emin=300;
  Emax=-10;
  
  DIAG=cvectorm(0,Nc-1);
  LOWDIAG=cvectorm(0,Nc-1);
  UPDIAG=cvectorm(0,Nc-1);
  ncarcnt=dvector(0,n*Nc-1);
  A1old=dmatrix(0,Nc-1,0,n-1);
  A2old=dmatrix(0,Nc-1,0,n-1); 

  for (i=0;i<Nc;i++)
    for (j=0;j<n;j++)
      {
	ix=j+i*n;
	A1old[i][j]=0;
	A2old[i][j]=0;
	ncarcnt[ix]=0;
      }

  passos=dvector(0,4*n-1);
  passod=dvector(0,4*n-1);
  for (i=0;i<4*n;i++)
    {
      passos[i]=0;
      passod[i]=0;
    }
  
  // Now I compute the energy range in which NEGF 
  // is solved if not specified by the user

  if ((Elower< -900)||(Eupper>900))

    for (i=0;i<n*Nc;i++)
      {
	if (Emax<=(-Phi[i]))
	  Emax=(-Phi[i]);
	if (Emin>=(-Phi[i]))
	  Emin=(-Phi[i]);
      }
  

  Emin=min(Emin,(mu2));
  Emin=min((mu1),Emin);
  Emax=max(Emax,(mu2));
  Emax=max((mu1),Emax);
  if (Eupper>900)
    Eupper=Emax+(Egap)*0.5+7*(vt);
  if (Elower<-900)
    Elower=Emin-(Egap)*0.5-7*(vt);


  if (!rank) printf("Eupper = %lg \n",Eupper);
  if (!rank) printf("Elower = %lg \n",Elower);
  if (!rank) printf("Current computed energy \n ");
  
  // NE is an extimation of the number of energy levels
  NE=(int)((Eupper-Elower)/dE);
  // ..... just to be sure to overestimate
  NE=NE+10;
  // Once known the number of energy levels I create the energy level and
  // the transmission coefficient matrix
  EE=dvector(0,NE-1);
  TE=dvector(0,NE-1);
  
  // I start to define the Hamiltonian for the carbon nanoribbon,
  // changed by sign
  
  // I create the Diagonal block changed by sign
  for (i=0;i<Nc;i++)
    {
      DIAG[i]=cmatrix(0,n-1,0,n-1);
      for (j=0;j<n;j++)
	for (k=0;k<n;k++)
	  DIAG[i][j][k]=zero;
      
      for (j=0;j<n;j++)
	DIAG[i][j][j]=complass(-(-Phi[j+i*n]),0);
    }
  
  
  // I create the Updiagonal block changed by sign
  for (i=0;i<(Nc-1);i++)
    UPDIAG[i]=create_updiagGNR(i,n,thop);
  
  UPDIAG[Nc-1]=cmatrix(0,n-1,0,n-1);
  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      UPDIAG[Nc-1][i][j]=zero;
  
  
  // I create the Lowdiag block changed by sign
  for (i=1;i<Nc;i++)
    LOWDIAG[i]=create_lowdiagGNR(i-1,n,thop);
  
  LOWDIAG[0]=cmatrix(0,n-1,0,n-1);
  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      LOWDIAG[0][i][j]=zero;


  // Now I take care of the diagonal to pass to the self-energy
  // I mean, the first and the last four unrolled CNT rings
  for (j=0;j<4;j++)
    for (i=0;i<n;i++)
      {
	passos[i+j*n]=DIAG[0][i][i].r;
	passod[i+j*n]=DIAG[Nc-1][i][i].r;
      }

  E=Elower;

  // The energy points are independent of one another, so instead of one
  // NEGF solve per energy they are handed to vides_rgf_batch() a chunk at
  // a time.  With a GPU present that chunk becomes the batch dimension of
  // the cuBLAS batched calls; without one the driver simply loops over the
  // chunk calling the same LDOS() as before, so the numbers are unchanged.
  {
    vides_rgf_desc vdesc;
    complex ***SSch,***SDch;
    double *Ech,*A1ch,*A2ch,*Tch;
    int NBmax,nb,ib;

    vdesc.n=n;
    vdesc.Nc=Nc;
    vdesc.NB=1;
    vdesc.variant=VIDES_RGF_STD;
    vdesc.flagtrans=1;
    vdesc.eta=eta;
    vdesc.Nreal=0;
    vdesc.order=NULL;

    NBmax=vides_negf_chunk(&vdesc);
    if (NBmax>NE) NBmax=NE;
    if (NBmax<1) NBmax=1;

    if (!rank) printf("NEGF backend: %s, %d energies per batch \n",
                      vides_gpu_describe(),NBmax);

    Ech =(double *)malloc((size_t)NBmax*sizeof(double));
    Tch =(double *)malloc((size_t)NBmax*sizeof(double));
    SSch=(complex ***)malloc((size_t)NBmax*sizeof(complex **));
    SDch=(complex ***)malloc((size_t)NBmax*sizeof(complex **));
    A1ch=(double *)malloc((size_t)NBmax*Nc*n*sizeof(double));
    A2ch=(double *)malloc((size_t)NBmax*Nc*n*sizeof(double));
    if (!Ech||!Tch||!SSch||!SDch||!A1ch||!A2ch)
      {
        printf("Out of memory allocating the NEGF energy batch \n");
        exit(0);
      }

    while (E<=(Eupper+dE*0.5))
      {
        // I fill one chunk of energies, building the Self-Energy for each
        // in case of Schottky barrier contacts or doped contacts
        nb=0;
        while ((nb<NBmax)&&(E<=(Eupper+dE*0.5)))
          {
            if (strcasecmp(GNRboundary,"doped")==0)
              {
                SSch[nb]=selfGNR(E,passos,n,thop,eta);
                SDch[nb]=selfGNR(E,passod,n,thop,eta);
              }
            else
              {
                SSch[nb]=selfschottky(E,(mu1),n,thop,eta);
                SDch[nb]=selfschottky(E,(mu2),n,thop,eta);
              }
            Ech[nb]=E;
            nb++;
            E+=dE;
          }

        vdesc.NB=nb;
        if (vides_rgf_batch(&vdesc,Ech,DIAG,UPDIAG,LOWDIAG,SSch,SDch,
                            A1ch,A2ch,Tch)!=0)
          {
            printf("NEGF batch failed \n");
            exit(0);
          }

        for (ib=0;ib<nb;ib++)
          {
            double Eb,*A1b,*A2b;
            Eb=Ech[ib];
            A1b=A1ch+(size_t)ib*Nc*n;
            A2b=A2ch+(size_t)ib*Nc*n;

            // I print out the energy level for which
            // the computation is performed every 50 energy levels
            if ((ie%50)==0)
              {
                if (!rank) printf("\r");
                if (!rank) printf("%lg ",Eb);
              }

            cfree_cmatrix(SSch[ib],0,n-1,0,n-1);
            cfree_cmatrix(SDch[ib],0,n-1,0,n-1);

            // I compute the charge
            for (i=0;i<Nc;i++)
      	for (j=0;j<n;j++)
      	  /* { */
      /* 	    // electrons */
      /* 	    ix=j+i*n; */
      /* 	    if ((A1b[i*n+j]>=1e-40)) */
      /* 	      if ((Eb>=-Phi[j+i*n])) */
      /* 		{ */
      /* 		  arg1=A1b[i*n+j]*Fermi_Dirac((Eb-(mu1))/(vt)); */
      /* 		  arg2=A1old[i][j]*Fermi_Dirac((E-dE-(mu1))/(vt)); */
      /* 		  ncarcnt[ix]=ncarcnt[ix] */
      /* 		    +2*(2*arg1)*dE/2; */
      /* 		} */
	    
      /* 	    if ((A2b[i*n+j]>=1e-40)) */
      /* 	      if ((Eb>=-Phi[j+i*n])) */
      /* 		{ */
      /* 		  arg1=A2b[i*n+j]*Fermi_Dirac((Eb-(mu2))/(vt)); */
      /* 		  arg2=A2old[i][j]*Fermi_Dirac((E-dE-(mu2))/(vt)); */
      /* 		  ncarcnt[ix]=ncarcnt[ix] */
      /* 		    +2*(2*arg1)*dE/2; */
      /* 		} */
	    
      /* 	    // last modifications */
      /* 	    //holes */
      /* 	    if ((A1b[i*n+j]>=1e-40)) */
      /* 	      if ((Eb<-Phi[j+i*n])) */
      /* 		{ */
      /* 		  arg1=A1b[i*n+j]*(1-Fermi_Dirac((Eb-(mu1))/(vt))); */
      /* 		  arg2=A1old[i][j]*(1-Fermi_Dirac((E-dE-(mu1))/(vt))); */
      /* 		  ncarcnt[ix]=ncarcnt[ix] */
      /* 		    -2*(2*arg1)*dE/2; */
      /* 		} */
	    
      /* 	    if ((A2b[i*n+j]>=1e-40)) */
      /* 	      if ((Eb<-Phi[j+i*n])) */
      /* 		{ */
      /* 		  arg1=A2b[i*n+j]*(1-Fermi_Dirac((Eb-(mu2))/(vt))); */
      /* 		  arg2=A2old[i][j]*(1-Fermi_Dirac((E-dE-(mu2))/(vt))); */
      /* 		  ncarcnt[ix]=ncarcnt[ix] */
      /* 		    -2*(2*arg1)*dE/2; */
      /* 		} */
      /* 	    // last modifications */
	    
      /* 	  } */

      	  {
      	    // electrons
      	    ix=j+i*n;
      	    if (A1b[i*n+j]>=1e-40)
      	      if ((Eb>=-Phi[j+i*n]))
      		{
      		  ncarcnt[ix]=ncarcnt[ix]
      		    -2*(
      			A1b[i*n+j]*Fermi_Dirac((Eb-(mu1))/(vt))*dE);
      		}
      	    if (A2b[i*n+j]>=1e-40)
      	      if ((Eb>=-Phi[j+i*n]))
      		{
      		  ncarcnt[ix]=ncarcnt[ix]
      		    -2*(
      			A2b[i*n+j]*Fermi_Dirac((Eb-(mu2))/(vt))*dE);
      		}
	    
      	    //holes
      	    if (A1b[i*n+j]>=1e-40)
      	      if ((Eb<-Phi[j+i*n]))
      		{
      		  ncarcnt[ix]=ncarcnt[ix]
      		    +2*(
      			A1b[i*n+j]*(1-Fermi_Dirac((Eb-(mu1))/(vt)))*dE);
      		}
	    
      	    if (A2b[i*n+j]>=1e-40)
      	      if ((Eb<-Phi[j+i*n]))
      		{
      		  ncarcnt[ix]=ncarcnt[ix]
      		    +2*(
      			A2b[i*n+j]*(1-Fermi_Dirac((Eb-(mu2))/(vt)))*dE);
      		}
      	    // last modifications
	    
      	  }
      

            EE[ie]=Eb;
            TE[ie]=Tch[ib];
            ie++;

            for (i=0;i<Nc;i++)
              for (j=0;j<n;j++)
                {
                  A1old[i][j]=A1b[i*n+j];
                  A2old[i][j]=A2b[i*n+j];
                }
          }
      }

    free(Ech);
    free(Tch);
    free(SSch);
    free(SDch);
    free(A1ch);
    free(A2ch);
  }

  if (!rank) printf("\n\n*********************************\n");
  if (!rank) printf("****** END OF NEGF in GNR *******\n"); 
  if (!rank) printf("*********************************\n\n");

  // I set the attribute of the class to send data out
  temp_obj=  PyArray_FromDimsAndData(1,&ie,PyArray_DOUBLE,(char*)EE);
  PyObject_SetAttrString(obj,"E",temp_obj);
  temp_obj=  PyArray_FromDimsAndData(1,&ie,PyArray_DOUBLE,(char*)TE);
  PyObject_SetAttrString(obj,"T",temp_obj);
  Nm=n*Nc;
  temp_obj=  PyArray_FromDimsAndData(1,&Nm,PyArray_DOUBLE,(char*)ncarcnt);
  PyObject_SetAttrString(obj,"charge",temp_obj);

  // Free the memory
  cfree_cvectorm(DIAG,0,Nc-1,0,n-1,0,n-1);
  cfree_cvectorm(LOWDIAG,0,Nc-1,0,n-1,0,n-1);
  cfree_cvectorm(UPDIAG,0,Nc-1,0,n-1,0,n-1);
  free_dmatrix(A1old,0,Nc-1,0,n-1);
  free_dmatrix(A2old,0,Nc-1,0,n-1);
  //free_dvector(EE,0,NE-1);
  //free_dvector(TE,0,NE-1);
  char *s = "Hello from GNR_charge_T";
  return Py_BuildValue("s", s);
}
