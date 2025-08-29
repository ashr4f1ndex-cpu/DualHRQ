#!/usr/bin/env python3
"""
Phase 1 Conditioning Agent - DualHRQ 2.0 Implementation
======================================================

CRITICAL PHASE: Conditioning Core + Budget Fix (Weeks 1-6)

This agent systematically implements the conditioning system following our
established TDD approach and 7-step development process.

Scope:
- Dynamic regime-based conditioning (replaces static puzzle_id)
- Parameter budget compliance enforcement
- Feature flag system for conditioning components
- Pattern library foundation
- Integration with HRM modules
- Comprehensive leakage validation

Success Gates:
- ≤27.5M parameters achieved
- Conditioning system ≤0.3M parameters  
- MI tests show no ID leakage
- Feature flags control all components
- Integration tests pass
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import yaml
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhaseStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"

@dataclass
class TaskResult:
    task_id: str
    status: PhaseStatus
    description: str
    details: Dict[str, Any]
    dependencies_met: bool = True
    next_actions: List[str] = None

class Phase1ConditioningAgent:
    """Phase 1 agent for systematic conditioning implementation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.phase_name = "Phase 1: Conditioning Core + Budget Fix"
        self.phase_duration = "Weeks 1-6"
        self.success_gates = {
            'parameter_budget': 27_500_000,
            'conditioning_budget': 300_000,
            'leakage_threshold': 0.1,  # bits
            'integration_tests_pass': True
        }
        
        # Task definitions from DRQ requirements
        self.tasks = {
            'DRQ-101': 'Pattern library foundation',
            'DRQ-102': 'RAG conditioning system', 
            'DRQ-103': 'Dynamic regime conditioning',
            'DRQ-104': 'HRM integration with conditioning',
            'DRQ-105': 'Leakage validation system',
            'DRQ-106': 'Feature engineering validation',
            'DRQ-107': 'Feature flags system',
            'DRQ-108': 'Budget compliance enforcement',
            'DRQ-109': 'Integration testing framework'
        }
        
        self.progress = {task_id: PhaseStatus.NOT_STARTED for task_id in self.tasks.keys()}
        
    def execute_phase(self) -> Dict[str, TaskResult]:
        """Execute Phase 1 following the 7-step development process."""
        logger.info(f"🚀 Starting {self.phase_name} ({self.phase_duration})")
        logger.info(f"Success Gates: {self.success_gates}")
        
        results = {}
        
        # Step 1: Parameter Budget Foundation (DRQ-108)
        results['DRQ-108'] = self._task_budget_compliance()
        if results['DRQ-108'].status == PhaseStatus.FAILED:
            logger.error("❌ CRITICAL: Budget compliance failed - cannot proceed")
            return results
            
        # Step 2: Feature Flags System (DRQ-107)  
        results['DRQ-107'] = self._task_feature_flags()
        
        # Step 3: Pattern Library Foundation (DRQ-101)
        results['DRQ-101'] = self._task_pattern_library()
        
        # Step 4: RAG System Implementation (DRQ-102)
        results['DRQ-102'] = self._task_rag_system()
        
        # Step 5: Dynamic Conditioning (DRQ-103)
        results['DRQ-103'] = self._task_dynamic_conditioning()
        
        # Step 6: HRM Integration (DRQ-104)
        results['DRQ-104'] = self._task_hrm_integration()
        
        # Step 7: Validation Systems (DRQ-105, DRQ-106, DRQ-109)
        results['DRQ-105'] = self._task_leakage_validation()
        results['DRQ-106'] = self._task_feature_validation()
        results['DRQ-109'] = self._task_integration_testing()
        
        # Phase completion check
        self._validate_phase_completion(results)
        
        return results
    
    def _task_budget_compliance(self) -> TaskResult:
        """DRQ-108: Budget compliance enforcement."""
        logger.info("🎯 DRQ-108: Implementing budget compliance enforcement")
        
        try:
            # Verify compliant configuration exists
            compliant_config = self.project_root / "config/compliant_hrm27m.yaml"
            if not compliant_config.exists():
                return TaskResult(
                    task_id='DRQ-108',
                    status=PhaseStatus.FAILED,
                    description='Compliant configuration not found',
                    details={'error': f'File not found: {compliant_config}'}
                )
            
            # Test parameter counting tool
            param_check = self._run_command([
                'python3', 'tools/param_count.py', 
                '--config', str(compliant_config),
                '--strict', '--format', 'json'
            ])
            
            if param_check['returncode'] != 0:
                return TaskResult(
                    task_id='DRQ-108',
                    status=PhaseStatus.FAILED, 
                    description='Parameter compliance check failed',
                    details={'error': param_check['stderr'], 'stdout': param_check['stdout']}
                )
            
            # Parse parameter count results
            try:
                param_data = json.loads(param_check['stdout'])
                total_params = param_data['total']
                is_compliant = param_data['is_compliant']
                
                if not is_compliant:
                    return TaskResult(
                        task_id='DRQ-108',
                        status=PhaseStatus.FAILED,
                        description='Configuration not within parameter budget',
                        details={'param_count': total_params, 'budget': self.success_gates['parameter_budget']}
                    )
                
                logger.info(f"✅ Parameter compliance verified: {total_params:,} parameters")
                
                return TaskResult(
                    task_id='DRQ-108',
                    status=PhaseStatus.COMPLETED,
                    description='Budget compliance enforcement operational',
                    details={
                        'param_count': total_params,
                        'budget_utilization': param_data.get('budget_utilization', 0),
                        'compliant': is_compliant
                    }
                )
                
            except json.JSONDecodeError as e:
                return TaskResult(
                    task_id='DRQ-108',
                    status=PhaseStatus.FAILED,
                    description='Failed to parse parameter count results',
                    details={'error': str(e), 'stdout': param_check['stdout']}
                )
                
        except Exception as e:
            logger.error(f"❌ DRQ-108 failed: {e}")
            return TaskResult(
                task_id='DRQ-108',
                status=PhaseStatus.FAILED,
                description=f'Budget compliance implementation failed: {e}',
                details={'exception': str(e)}
            )
    
    def _task_feature_flags(self) -> TaskResult:
        """DRQ-107: Feature flags system implementation."""
        logger.info("🚩 DRQ-107: Implementing feature flags system")
        
        try:
            # Check if feature flags module exists and is functional
            feature_flags_path = self.project_root / "src/models/conditioning/feature_flags.py"
            
            if not feature_flags_path.exists():
                return TaskResult(
                    task_id='DRQ-107',
                    status=PhaseStatus.BLOCKED,
                    description='Feature flags module not found',
                    details={'required_file': str(feature_flags_path)},
                    next_actions=['Create feature_flags.py module', 'Implement FeatureFlags class']
                )
            
            # Run feature flags test
            test_result = self._run_command([
                'python3', '-m', 'pytest', 
                'tests/conditioning/test_feature_flags.py', 
                '-v'
            ])
            
            if test_result['returncode'] == 0:
                logger.info("✅ Feature flags system operational")
                return TaskResult(
                    task_id='DRQ-107',
                    status=PhaseStatus.COMPLETED,
                    description='Feature flags system implemented and tested',
                    details={'test_output': test_result['stdout']}
                )
            else:
                return TaskResult(
                    task_id='DRQ-107',
                    status=PhaseStatus.IN_PROGRESS,
                    description='Feature flags tests failing',
                    details={'test_errors': test_result['stderr']},
                    next_actions=['Fix failing tests', 'Implement missing functionality']
                )
                
        except Exception as e:
            logger.error(f"❌ DRQ-107 failed: {e}")
            return TaskResult(
                task_id='DRQ-107',
                status=PhaseStatus.FAILED,
                description=f'Feature flags implementation failed: {e}',
                details={'exception': str(e)}
            )
    
    def _task_pattern_library(self) -> TaskResult:
        """DRQ-101: Pattern library foundation."""
        logger.info("📚 DRQ-101: Implementing pattern library foundation")
        
        try:
            # Check pattern library implementation
            pattern_lib_path = self.project_root / "src/models/pattern_library.py"
            
            if not pattern_lib_path.exists():
                return TaskResult(
                    task_id='DRQ-101',
                    status=PhaseStatus.BLOCKED,
                    description='Pattern library module not found',
                    details={'required_file': str(pattern_lib_path)},
                    next_actions=['Create pattern_library.py', 'Implement PatternLibrary class']
                )
            
            # Test pattern library
            test_result = self._run_command([
                'python3', '-m', 'pytest',
                'tests/conditioning/test_pattern_library.py',
                '-v'
            ])
            
            if test_result['returncode'] == 0:
                logger.info("✅ Pattern library foundation operational")
                return TaskResult(
                    task_id='DRQ-101',
                    status=PhaseStatus.COMPLETED,
                    description='Pattern library foundation implemented',
                    details={'test_output': test_result['stdout']}
                )
            else:
                return TaskResult(
                    task_id='DRQ-101',
                    status=PhaseStatus.IN_PROGRESS,
                    description='Pattern library tests failing',
                    details={'test_errors': test_result['stderr']},
                    next_actions=['Fix pattern library implementation', 'Resolve test failures']
                )
                
        except Exception as e:
            logger.error(f"❌ DRQ-101 failed: {e}")
            return TaskResult(
                task_id='DRQ-101',
                status=PhaseStatus.FAILED,
                description=f'Pattern library implementation failed: {e}',
                details={'exception': str(e)}
            )
    
    def _task_rag_system(self) -> TaskResult:
        """DRQ-102: RAG conditioning system."""
        logger.info("🔍 DRQ-102: Implementing RAG conditioning system")
        
        try:
            # Check RAG system implementation
            rag_path = self.project_root / "src/models/conditioning/rag_system.py"
            
            if not rag_path.exists():
                return TaskResult(
                    task_id='DRQ-102',
                    status=PhaseStatus.BLOCKED,
                    description='RAG system module not found',
                    details={'required_file': str(rag_path)},
                    next_actions=['Create rag_system.py', 'Implement RAGSystem class']
                )
            
            # Test RAG system
            test_result = self._run_command([
                'python3', '-m', 'pytest',
                'tests/conditioning/test_rag_system.py',
                '-v'
            ])
            
            if test_result['returncode'] == 0:
                logger.info("✅ RAG system operational")
                return TaskResult(
                    task_id='DRQ-102',
                    status=PhaseStatus.COMPLETED,
                    description='RAG conditioning system implemented',
                    details={'test_output': test_result['stdout']}
                )
            else:
                return TaskResult(
                    task_id='DRQ-102',
                    status=PhaseStatus.IN_PROGRESS,
                    description='RAG system tests failing',
                    details={'test_errors': test_result['stderr']},
                    next_actions=['Implement RAG retrieval', 'Fix conditioning integration']
                )
                
        except Exception as e:
            logger.error(f"❌ DRQ-102 failed: {e}")
            return TaskResult(
                task_id='DRQ-102',
                status=PhaseStatus.FAILED,
                description=f'RAG system implementation failed: {e}',
                details={'exception': str(e)}
            )
    
    def _task_dynamic_conditioning(self) -> TaskResult:
        """DRQ-103: Dynamic regime conditioning."""
        logger.info("⚡ DRQ-103: Implementing dynamic regime conditioning")
        
        try:
            # Check regime features implementation
            regime_path = self.project_root / "src/models/conditioning/regime_features.py"
            
            if not regime_path.exists():
                return TaskResult(
                    task_id='DRQ-103',
                    status=PhaseStatus.BLOCKED,
                    description='Regime features module not found',
                    details={'required_file': str(regime_path)},
                    next_actions=['Create regime_features.py', 'Implement RegimeFeatures class']
                )
            
            # Test no static IDs
            static_id_check = self._run_command([
                'python3', 'tools/check_no_static_ids.py'
            ])
            
            if static_id_check['returncode'] != 0:
                return TaskResult(
                    task_id='DRQ-103',
                    status=PhaseStatus.FAILED,
                    description='Static puzzle_id usage detected',
                    details={'violations': static_id_check['stdout']},
                    next_actions=['Remove static puzzle_id references', 'Implement dynamic conditioning']
                )
            
            logger.info("✅ Dynamic conditioning system operational")
            return TaskResult(
                task_id='DRQ-103',
                status=PhaseStatus.COMPLETED,
                description='Dynamic regime conditioning implemented',
                details={'static_id_check': 'PASSED'}
            )
            
        except Exception as e:
            logger.error(f"❌ DRQ-103 failed: {e}")
            return TaskResult(
                task_id='DRQ-103',
                status=PhaseStatus.FAILED,
                description=f'Dynamic conditioning implementation failed: {e}',
                details={'exception': str(e)}
            )
    
    def _task_hrm_integration(self) -> TaskResult:
        """DRQ-104: HRM integration with conditioning."""
        logger.info("🔗 DRQ-104: Implementing HRM integration")
        
        try:
            # Test HRM integration
            test_result = self._run_command([
                'python3', '-m', 'pytest',
                'tests/conditioning/test_hrm_integration.py',
                '-v'
            ])
            
            if test_result['returncode'] == 0:
                logger.info("✅ HRM integration operational")
                return TaskResult(
                    task_id='DRQ-104',
                    status=PhaseStatus.COMPLETED,
                    description='HRM integration with conditioning complete',
                    details={'test_output': test_result['stdout']}
                )
            else:
                return TaskResult(
                    task_id='DRQ-104',
                    status=PhaseStatus.IN_PROGRESS,
                    description='HRM integration tests failing',
                    details={'test_errors': test_result['stderr']},
                    next_actions=['Fix HRM conditioning integration', 'Resolve test failures']
                )
                
        except Exception as e:
            logger.error(f"❌ DRQ-104 failed: {e}")
            return TaskResult(
                task_id='DRQ-104',
                status=PhaseStatus.FAILED,
                description=f'HRM integration failed: {e}',
                details={'exception': str(e)}
            )
    
    def _task_leakage_validation(self) -> TaskResult:
        """DRQ-105: Leakage validation system."""
        logger.info("🕵️ DRQ-105: Implementing leakage validation")
        
        try:
            # Test leakage validation
            test_result = self._run_command([
                'python3', '-m', 'pytest',
                'tests/conditioning/test_leakage_mi.py',
                '-v'
            ])
            
            if test_result['returncode'] == 0:
                logger.info("✅ Leakage validation operational")
                return TaskResult(
                    task_id='DRQ-105',
                    status=PhaseStatus.COMPLETED,
                    description='Leakage validation system implemented',
                    details={'test_output': test_result['stdout']}
                )
            else:
                return TaskResult(
                    task_id='DRQ-105',
                    status=PhaseStatus.IN_PROGRESS,
                    description='Leakage validation tests failing',
                    details={'test_errors': test_result['stderr']},
                    next_actions=['Implement MI calculation', 'Fix leakage detection']
                )
                
        except Exception as e:
            logger.error(f"❌ DRQ-105 failed: {e}")
            return TaskResult(
                task_id='DRQ-105',
                status=PhaseStatus.FAILED,
                description=f'Leakage validation failed: {e}',
                details={'exception': str(e)}
            )
    
    def _task_feature_validation(self) -> TaskResult:
        """DRQ-106: Feature engineering validation."""
        logger.info("🔬 DRQ-106: Implementing feature validation")
        
        try:
            # Test shuffle codes (feature validation)
            test_result = self._run_command([
                'python3', '-m', 'pytest',
                'tests/conditioning/test_shuffle_codes.py',
                '-v'
            ])
            
            if test_result['returncode'] == 0:
                logger.info("✅ Feature validation operational")
                return TaskResult(
                    task_id='DRQ-106',
                    status=PhaseStatus.COMPLETED,
                    description='Feature engineering validation implemented',
                    details={'test_output': test_result['stdout']}
                )
            else:
                return TaskResult(
                    task_id='DRQ-106',
                    status=PhaseStatus.IN_PROGRESS,
                    description='Feature validation tests failing',
                    details={'test_errors': test_result['stderr']},
                    next_actions=['Implement shuffle validation', 'Fix feature tests']
                )
                
        except Exception as e:
            logger.error(f"❌ DRQ-106 failed: {e}")
            return TaskResult(
                task_id='DRQ-106',
                status=PhaseStatus.FAILED,
                description=f'Feature validation failed: {e}',
                details={'exception': str(e)}
            )
    
    def _task_integration_testing(self) -> TaskResult:
        """DRQ-109: Integration testing framework."""
        logger.info("🧪 DRQ-109: Implementing integration testing")
        
        try:
            # Test parameter gating system
            test_result = self._run_command([
                'python3', '-m', 'pytest',
                'tests/conditioning/test_param_gate.py',
                '-v'
            ])
            
            if test_result['returncode'] == 0:
                logger.info("✅ Integration testing operational")
                return TaskResult(
                    task_id='DRQ-109',
                    status=PhaseStatus.COMPLETED,
                    description='Integration testing framework implemented',
                    details={'test_output': test_result['stdout']}
                )
            else:
                return TaskResult(
                    task_id='DRQ-109',
                    status=PhaseStatus.IN_PROGRESS,
                    description='Integration tests failing',
                    details={'test_errors': test_result['stderr']},
                    next_actions=['Fix integration test framework', 'Resolve test dependencies']
                )
                
        except Exception as e:
            logger.error(f"❌ DRQ-109 failed: {e}")
            return TaskResult(
                task_id='DRQ-109',
                status=PhaseStatus.FAILED,
                description=f'Integration testing failed: {e}',
                details={'exception': str(e)}
            )
    
    def _validate_phase_completion(self, results: Dict[str, TaskResult]) -> bool:
        """Validate Phase 1 completion against success gates."""
        logger.info("🎯 Validating Phase 1 completion against success gates")
        
        # Count completed vs failed tasks
        completed_tasks = [t for t in results.values() if t.status == PhaseStatus.COMPLETED]
        failed_tasks = [t for t in results.values() if t.status == PhaseStatus.FAILED]
        in_progress_tasks = [t for t in results.values() if t.status == PhaseStatus.IN_PROGRESS]
        
        logger.info(f"Task Summary: {len(completed_tasks)} completed, {len(failed_tasks)} failed, {len(in_progress_tasks)} in progress")
        
        # Check critical success gates
        if len(failed_tasks) > 0:
            logger.error(f"❌ Phase 1 FAILED: {len(failed_tasks)} critical tasks failed")
            for task in failed_tasks:
                logger.error(f"  - {task.task_id}: {task.description}")
            return False
        
        if len(completed_tasks) == len(self.tasks):
            logger.info("✅ Phase 1 COMPLETED: All tasks successful")
            return True
        else:
            logger.warning(f"⚠️ Phase 1 IN PROGRESS: {len(completed_tasks)}/{len(self.tasks)} tasks completed")
            return False
    
    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run command and return result."""
        if cwd is None:
            cwd = self.project_root
            
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'cmd': ' '.join(cmd)
            }
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'cmd': ' '.join(cmd)
            }
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'cmd': ' '.join(cmd)
            }

def main():
    """Main entry point for Phase 1 agent."""
    project_root = Path(__file__).parent.parent
    
    logger.info("🤖 Phase 1 Conditioning Agent - DualHRQ 2.0")
    logger.info("=" * 60)
    
    agent = Phase1ConditioningAgent(project_root)
    results = agent.execute_phase()
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("PHASE 1 EXECUTION SUMMARY")
    print("=" * 60)
    
    for task_id, result in results.items():
        status_icon = {
            PhaseStatus.COMPLETED: "✅",
            PhaseStatus.IN_PROGRESS: "⚠️",
            PhaseStatus.FAILED: "❌",
            PhaseStatus.BLOCKED: "🚫",
            PhaseStatus.NOT_STARTED: "⏸️"
        }[result.status]
        
        print(f"{status_icon} {task_id}: {result.description}")
        if result.next_actions:
            for action in result.next_actions:
                print(f"    → {action}")
    
    # Determine overall phase status
    completed = len([r for r in results.values() if r.status == PhaseStatus.COMPLETED])
    total = len(results)
    
    print(f"\nOverall Progress: {completed}/{total} tasks completed")
    
    if completed == total:
        print("🎉 PHASE 1 COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print(f"⚠️ PHASE 1 IN PROGRESS ({completed}/{total})")
        return 1

if __name__ == '__main__':
    sys.exit(main())