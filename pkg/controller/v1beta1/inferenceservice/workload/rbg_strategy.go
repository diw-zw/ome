package workload

import (
	"context"
	"fmt"

	"github.com/go-logr/logr"
	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/sgl-project/ome/pkg/apis/ome/v1beta1"
	"github.com/sgl-project/ome/pkg/constants"
	"github.com/sgl-project/ome/pkg/controller/v1beta1/inferenceservice/components"
	"github.com/sgl-project/ome/pkg/controller/v1beta1/inferenceservice/reconcilers/hpa"
	"github.com/sgl-project/ome/pkg/controller/v1beta1/inferenceservice/reconcilers/rbac"
	"github.com/sgl-project/ome/pkg/controller/v1beta1/inferenceservice/reconcilers/rbg"
)

// RBGStrategyName is the registered name of the RBG workload strategy.
const RBGStrategyName = "RBG"

// RBGStrategy implements WorkloadStrategy by packing every InferenceService
// component into a single RoleBasedGroup CR. It is the All-in-One workload
// path for OME and supports RawDeployment and MultiNode per-component
// deployment modes.
//
// Resources reconciled per call:
//   - One RoleBasedGroup with a role per component
//   - One ServiceAccount (and Role/RoleBinding for the Router component)
//     per component, mirroring SingleComponentStrategy
//   - One HPA per role to retain independent scaling behaviour
type RBGStrategy struct {
	Client    client.Client
	Clientset kubernetes.Interface
	Scheme    *runtime.Scheme
	Log       logr.Logger
}

// NewRBGStrategy constructs an RBGStrategy.
func NewRBGStrategy(c client.Client, clientset kubernetes.Interface, scheme *runtime.Scheme, log logr.Logger) *RBGStrategy {
	return &RBGStrategy{
		Client:    c,
		Clientset: clientset,
		Scheme:    scheme,
		Log:       log.WithName("RBGStrategy"),
	}
}

// GetStrategyName returns the registration name.
func (s *RBGStrategy) GetStrategyName() string { return RBGStrategyName }

// IsApplicable returns true when the InferenceService selects the
// RoleBasedGroup deployment mode via annotation / config.
func (s *RBGStrategy) IsApplicable(_ *v1beta1.InferenceService, deploymentMode constants.DeploymentModeType) bool {
	return deploymentMode == constants.RoleBasedGroup
}

// ValidateDeploymentModes restricts per-component deployment modes to those
// the underlying RBG roles can express. Serverless / MultiNodeRayVLLM /
// VirtualDeployment are explicitly unsupported in the Alpha RBG strategy.
func (s *RBGStrategy) ValidateDeploymentModes(modes *ComponentDeploymentModes) error {
	if modes == nil {
		return nil
	}
	if err := validateRBGComponentMode("engine", modes.Engine, false); err != nil {
		return err
	}
	if err := validateRBGComponentMode("decoder", modes.Decoder, true); err != nil {
		return err
	}
	if err := validateRBGComponentMode("router", modes.Router, true); err != nil {
		return err
	}
	return nil
}

func validateRBGComponentMode(name string, mode constants.DeploymentModeType, optional bool) error {
	if mode == "" {
		if optional {
			return nil
		}
		return fmt.Errorf("RBG strategy requires a deployment mode for %s", name)
	}
	switch mode {
	case constants.RawDeployment, constants.MultiNode:
		return nil
	default:
		return fmt.Errorf("RBG strategy does not support deployment mode %q for %s (only RawDeployment and MultiNode are supported)", mode, name)
	}
}

// ReconcileWorkload extracts per-component RoleConfigs, then reconciles
// RBAC, the RoleBasedGroup CR and per-role HPAs.
func (s *RBGStrategy) ReconcileWorkload(ctx context.Context, request *WorkloadReconcileRequest) (ctrl.Result, error) {
	if request == nil || request.InferenceService == nil {
		return ctrl.Result{}, errors.New("RBG strategy received empty reconcile request")
	}
	isvc := request.InferenceService
	s.Log.Info("Reconciling with RBG strategy", "namespace", isvc.Namespace, "inferenceService", isvc.Name)

	configs, err := s.extractRoleConfigs(request)
	if err != nil {
		return ctrl.Result{}, err
	}
	if len(configs) == 0 {
		return ctrl.Result{}, errors.New("RBG strategy requires at least one component (engine, decoder or router) to be defined")
	}

	if err := s.reconcileRBAC(isvc, configs); err != nil {
		return ctrl.Result{}, err
	}

	rbgRec := rbg.NewRBGReconciler(s.Client, s.Scheme, s.Log)
	if result, err := rbgRec.Reconcile(ctx, isvc, configs); err != nil {
		return result, err
	} else if result.Requeue || result.RequeueAfter > 0 {
		return result, nil
	}

	if err := s.reconcileHPAs(isvc, configs); err != nil {
		return ctrl.Result{}, err
	}

	s.Log.Info("RBG strategy reconciliation completed", "namespace", isvc.Namespace, "inferenceService", isvc.Name)
	return ctrl.Result{}, nil
}

// extractRoleConfigs builds RoleConfigs for every component the
// InferenceService defines, in a stable order so the resulting RBG roles
// list is deterministic across reconciles.
func (s *RBGStrategy) extractRoleConfigs(request *WorkloadReconcileRequest) ([]*components.RoleConfig, error) {
	if request.ComponentBuilderFactory == nil {
		return nil, errors.New("RBG strategy requires a non-nil ComponentBuilderFactory")
	}
	if request.DeploymentModes == nil {
		return nil, errors.New("RBG strategy requires non-nil DeploymentModes")
	}
	configs := make([]*components.RoleConfig, 0, 3)

	if request.MergedEngine != nil {
		comp := request.ComponentBuilderFactory.CreateEngineComponent(
			request.DeploymentModes.Engine,
			request.BaseModel,
			request.BaseModelMeta,
			request.MergedEngine,
			request.Runtime,
			request.RuntimeName,
			request.EngineSupportedModelFormat,
			request.EngineAcceleratorClass,
			request.EngineAcceleratorClassName,
		)
		cfg, err := extractFromComponent(comp, request.InferenceService, "engine")
		if err != nil {
			return nil, err
		}
		configs = append(configs, cfg)
	}

	if request.MergedDecoder != nil {
		comp := request.ComponentBuilderFactory.CreateDecoderComponent(
			request.DeploymentModes.Decoder,
			request.BaseModel,
			request.BaseModelMeta,
			request.MergedDecoder,
			request.Runtime,
			request.RuntimeName,
			request.DecoderSupportedModelFormat,
			request.DecoderAcceleratorClass,
			request.DecoderAcceleratorClassName,
		)
		cfg, err := extractFromComponent(comp, request.InferenceService, "decoder")
		if err != nil {
			return nil, err
		}
		configs = append(configs, cfg)
	}

	if request.MergedRouter != nil {
		comp := request.ComponentBuilderFactory.CreateRouterComponent(
			request.DeploymentModes.Router,
			request.BaseModel,
			request.BaseModelMeta,
			request.MergedRouter,
			request.Runtime,
			request.RuntimeName,
		)
		cfg, err := extractFromComponent(comp, request.InferenceService, "router")
		if err != nil {
			return nil, err
		}
		configs = append(configs, cfg)
	}

	return configs, nil
}

func extractFromComponent(comp components.Component, isvc *v1beta1.InferenceService, name string) (*components.RoleConfig, error) {
	extractor, ok := comp.(components.RoleConfigExtractor)
	if !ok {
		return nil, fmt.Errorf("component %s does not implement RoleConfigExtractor and cannot be used with the RBG strategy", name)
	}
	cfg, err := extractor.ExtractRoleConfig(isvc)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to extract %s role config", name)
	}
	return cfg, nil
}

// reconcileRBAC mirrors the RBAC handling done by SingleComponentStrategy.
// Only Router currently needs a Role / RoleBinding, but every component
// gets a ServiceAccount so pods can use a stable identity.
func (s *RBGStrategy) reconcileRBAC(isvc *v1beta1.InferenceService, configs []*components.RoleConfig) error {
	for _, cfg := range configs {
		rbacRec := rbac.NewRBACReconciler(s.Client, s.Scheme, cfg.ObjectMeta, cfg.ComponentType, isvc)
		if err := rbacRec.Reconcile(); err != nil {
			return errors.Wrapf(err, "failed to reconcile RBAC for %s", cfg.ComponentType)
		}
		// Bind the created ServiceAccount to every pod spec on this role so
		// that Router (and any future component needing Role/RoleBinding)
		// actually runs under its dedicated identity rather than the default
		// ServiceAccount of the namespace. SingleComponentStrategy does this
		// at the component level via router.go; RBG packs roles into one CR
		// and therefore has to propagate the name itself.
		saName := rbacRec.GetServiceAccountName()
		if cfg.PodSpec != nil {
			cfg.PodSpec.ServiceAccountName = saName
		}
		if cfg.LeaderPodSpec != nil {
			cfg.LeaderPodSpec.ServiceAccountName = saName
		}
		if cfg.WorkerPodSpec != nil {
			cfg.WorkerPodSpec.ServiceAccountName = saName
		}
	}
	return nil
}

// reconcileHPAs creates one HPA per role so each component scales
// independently — matching SingleComponentStrategy semantics. The HPA's
// scale target is the underlying workload created by the RBG controller.
// In v1alpha2 of the RBG API this is a Deployment for RawDeployment roles.
// MultiNode roles intentionally have no HPA: scaling LeaderWorkerSet groups
// is owned by the RBG ScalingAdapter, not OME.
func (s *RBGStrategy) reconcileHPAs(isvc *v1beta1.InferenceService, configs []*components.RoleConfig) error {
	for _, cfg := range configs {
		if cfg.DeploymentMode != constants.RawDeployment {
			continue
		}
		if cfg.ComponentExtensionSpec == nil {
			continue
		}
		hpaRec := hpa.NewHPAReconciler(s.Client, s.Scheme, cfg.ObjectMeta, cfg.ComponentExtensionSpec)
		if err := hpaRec.SetControllerReferences(isvc, s.Scheme); err != nil {
			return errors.Wrapf(err, "failed to set HPA owner reference for %s", cfg.ComponentType)
		}
		if _, err := hpaRec.Reconcile(); err != nil {
			return errors.Wrapf(err, "failed to reconcile HPA for %s", cfg.ComponentType)
		}
	}
	return nil
}

// Compile-time interface check.
var _ WorkloadStrategy = (*RBGStrategy)(nil)
