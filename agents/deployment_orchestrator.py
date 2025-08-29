#!/usr/bin/env python3
"""
DualHRQ 2.0 Deployment Orchestrator
==================================

SYSTEMATIC AGENT DEPLOYMENT FOR DUALHRQ 2.0 IMPLEMENTATION

This orchestrator manages the systematic deployment of agents across all 4 phases
of the DualHRQ 2.0 implementation following our established plan:

Phase 1: Conditioning Core + Budget Fix (Weeks 1-6)
Phase 2: Pattern Library + HRM Integration (Weeks 7-12)  
Phase 3: Enhanced Conditioning + Validation (Weeks 13-18)
Phase 4: Agents + Infrastructure (Weeks 19-26)

Key Features:
- Follows 7-step TDD development process
- Enforces parameter budget compliance (26.5M-27.5M)
- Comprehensive validation and testing
- Zero regulatory violations (SSR/LULD)
- Production-ready deployment

Success Criteria:
- Parameter budget: 26.5M ≤ total ≤ 27.5M
- Statistical significance: RC p<0.05, SPA p<0.10
- Performance: Sharpe >1.5, Max DD <15%, Latency <100ms  
- Regulatory: Zero SSR/LULD violations
- Operational: >99.9% uptime, <30s kill switch
"""

import sys
import os
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import subprocess
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class DeploymentPhase(Enum):
    PHASE_0_FOUNDATION = "phase_0_foundation"
    PHASE_1_CONDITIONING = "phase_1_conditioning" 
    PHASE_2_PATTERN_HRM = "phase_2_pattern_hrm"
    PHASE_3_VALIDATION = "phase_3_validation"
    PHASE_4_PRODUCTION = "phase_4_production"

class PhaseStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"

@dataclass
class PhaseResult:
    phase: DeploymentPhase
    status: PhaseStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[timedelta]
    tasks_completed: int
    tasks_total: int
    success_gates_met: Dict[str, bool]
    next_actions: List[str]
    blocking_issues: List[str]
    details: Dict[str, Any]

@dataclass 
class DeploymentPlan:
    """DualHRQ 2.0 deployment plan with phase details."""
    project_duration_weeks: int = 26
    team_size: str = "5-6 engineers"
    methodology: str = "Test-Driven Development with 7-step process"
    parameter_budget_min: int = 26_500_000
    parameter_budget_max: int = 27_500_000
    
    phases: Dict[DeploymentPhase, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.phases is None:
            self.phases = {
                DeploymentPhase.PHASE_0_FOUNDATION: {
                    'name': 'Foundation & Guards',
                    'weeks': '1-2',
                    'focus': 'Parameter gates, ID detection, determinism',
                    'success_gates': ['CI gates operational', 'Static ID detection', 'Determinism tests'],
                    'critical_path': True,
                    'dependencies': []
                },
                DeploymentPhase.PHASE_1_CONDITIONING: {
                    'name': 'Conditioning + Budget',
                    'weeks': '3-8', 
                    'focus': 'Dynamic conditioning, parameter compliance',
                    'success_gates': ['≤27.5M params', '<100ms latency', 'No leakage'],
                    'critical_path': True,
                    'dependencies': [DeploymentPhase.PHASE_0_FOUNDATION]
                },
                DeploymentPhase.PHASE_2_PATTERN_HRM: {
                    'name': 'Pattern & HRM Integration',
                    'weeks': '9-14',
                    'focus': 'Advanced patterns, walk-forward validation',
                    'success_gates': ['10K+ patterns', 'No leakage', 'Integration tests pass'],
                    'critical_path': True,
                    'dependencies': [DeploymentPhase.PHASE_1_CONDITIONING]
                },
                DeploymentPhase.PHASE_3_VALIDATION: {
                    'name': 'Validation & Enhancement',
                    'weeks': '15-20',
                    'focus': 'Statistical tests, regulatory compliance',
                    'success_gates': ['RC/SPA/DSR pass', 'Zero violations', 'Performance certified'],
                    'critical_path': True,
                    'dependencies': [DeploymentPhase.PHASE_2_PATTERN_HRM]
                },
                DeploymentPhase.PHASE_4_PRODUCTION: {
                    'name': 'Production & Agents',
                    'weeks': '21-26',
                    'focus': 'Agent system, deployment, go-live',
                    'success_gates': ['Production operational', '<30s kill switch', 'Monitoring active'],
                    'critical_path': True,
                    'dependencies': [DeploymentPhase.PHASE_3_VALIDATION]
                }
            }

class DeploymentOrchestrator:
    """Orchestrates systematic DualHRQ 2.0 deployment across all phases."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.deployment_plan = DeploymentPlan()
        self.deployment_state = self._load_deployment_state()
        
        # Initialize agents directory
        self.agents_dir = project_root / "agents"
        self.agents_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 DualHRQ 2.0 Deployment Orchestrator Initialized")
        logger.info(f"Project Root: {project_root}")
        logger.info(f"Parameter Budget: {self.deployment_plan.parameter_budget_min:,} - {self.deployment_plan.parameter_budget_max:,}")
    
    def deploy_systematic_implementation(self) -> Dict[DeploymentPhase, PhaseResult]:
        """Deploy the complete DualHRQ 2.0 system systematically."""
        logger.info("🎯 Starting systematic DualHRQ 2.0 deployment")
        logger.info("=" * 80)
        
        results = {}
        
        # Deploy each phase in order
        for phase in DeploymentPhase:
            logger.info(f"📋 Preparing {phase.value}")
            
            # Check dependencies
            if not self._check_phase_dependencies(phase, results):
                logger.error(f"❌ {phase.value} blocked by failed dependencies")
                results[phase] = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.BLOCKED,
                    start_time=datetime.now(),
                    end_time=None,
                    duration=None,
                    tasks_completed=0,
                    tasks_total=0,
                    success_gates_met={},
                    next_actions=['Resolve dependency failures'],
                    blocking_issues=['Dependencies not met'],
                    details={'dependencies': self.deployment_plan.phases[phase]['dependencies']}
                )
                break
            
            # Execute phase
            result = self._execute_phase(phase)
            results[phase] = result
            
            # Check if phase failed
            if result.status == PhaseStatus.FAILED:
                logger.error(f"❌ {phase.value} FAILED - stopping deployment")
                break
                
            # Update deployment state
            self._save_deployment_state(results)
        
        # Generate final deployment report
        self._generate_deployment_report(results)
        
        return results
    
    def _execute_phase(self, phase: DeploymentPhase) -> PhaseResult:
        """Execute a specific deployment phase."""
        phase_info = self.deployment_plan.phases[phase]
        start_time = datetime.now()
        
        logger.info(f"🚀 Executing {phase.value}")
        logger.info(f"   Name: {phase_info['name']}")
        logger.info(f"   Weeks: {phase_info['weeks']}")
        logger.info(f"   Focus: {phase_info['focus']}")
        logger.info(f"   Success Gates: {phase_info['success_gates']}")
        
        try:
            if phase == DeploymentPhase.PHASE_0_FOUNDATION:
                return self._execute_phase_0_foundation(phase, start_time)
            elif phase == DeploymentPhase.PHASE_1_CONDITIONING:
                return self._execute_phase_1_conditioning(phase, start_time)
            elif phase == DeploymentPhase.PHASE_2_PATTERN_HRM:
                return self._execute_phase_2_pattern_hrm(phase, start_time)
            elif phase == DeploymentPhase.PHASE_3_VALIDATION:
                return self._execute_phase_3_validation(phase, start_time)
            elif phase == DeploymentPhase.PHASE_4_PRODUCTION:
                return self._execute_phase_4_production(phase, start_time)
            else:
                raise ValueError(f"Unknown phase: {phase}")
                
        except Exception as e:
            logger.error(f"❌ Phase {phase.value} execution failed: {e}")
            return PhaseResult(
                phase=phase,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=datetime.now() - start_time,
                tasks_completed=0,
                tasks_total=0,
                success_gates_met={},
                next_actions=[f'Debug phase execution error: {e}'],
                blocking_issues=[str(e)],
                details={'exception': str(e)}
            )
    
    def _execute_phase_0_foundation(self, phase: DeploymentPhase, start_time: datetime) -> PhaseResult:
        """Execute Phase 0: Foundation & Guards."""
        logger.info("🏗️ Phase 0: Foundation & Guards")
        
        # Foundation tasks
        foundation_tasks = {
            'parameter_counter': 'Parameter counter + CI gate operational',
            'static_id_detection': 'Static ID detection prevents commits', 
            'determinism_tests': 'Determinism tests establish baseline',
            'import_stubs': 'All imports work, no CI failures'
        }
        
        completed_tasks = 0
        success_gates_met = {}
        next_actions = []
        blocking_issues = []
        
        # Task 1: Parameter Counter
        logger.info("🎯 Foundation Task: Parameter counter + CI gate")
        param_check_result = self._run_command([
            'python3', 'tools/param_count.py', 
            '--config', 'config/compliant_hrm27m.yaml', 
            '--strict'
        ])
        
        if param_check_result['returncode'] == 0:
            success_gates_met['parameter_counter'] = True
            completed_tasks += 1
            logger.info("✅ Parameter counter operational")
        else:
            success_gates_met['parameter_counter'] = False
            blocking_issues.append('Parameter counter failing')
            next_actions.append('Fix parameter counter implementation')
        
        # Task 2: Static ID Detection
        logger.info("🔍 Foundation Task: Static ID detection")
        static_id_result = self._run_command([
            'python3', 'tools/check_no_static_ids.py'
        ])
        
        if static_id_result['returncode'] == 0:
            success_gates_met['static_id_detection'] = True
            completed_tasks += 1
            logger.info("✅ Static ID detection operational")
        else:
            success_gates_met['static_id_detection'] = False
            blocking_issues.append('Static IDs still present in code')
            next_actions.append('Remove static puzzle_id references')
        
        # Task 3: Determinism Tests
        logger.info("🧪 Foundation Task: Determinism tests")
        determinism_result = self._run_command([
            'python3', '-m', 'pytest', 'tests/conditioning/test_determinism.py', '-v'
        ])
        
        if determinism_result['returncode'] == 0:
            success_gates_met['determinism_tests'] = True
            completed_tasks += 1
            logger.info("✅ Determinism tests operational")
        else:
            success_gates_met['determinism_tests'] = False
            blocking_issues.append('Determinism tests failing')
            next_actions.append('Fix determinism test implementation')
        
        # Task 4: Import Stubs (check key imports work)
        logger.info("📦 Foundation Task: Import validation")
        import_tests = [
            'lab_v10/tests/test_hrm_integration_smoke.py',
            'tests/conditioning/test_hrm_integration.py'
        ]
        
        import_success = True
        for test_file in import_tests:
            result = self._run_command([
                'python3', '-m', 'pytest', test_file, '--collect-only'
            ])
            if result['returncode'] != 0:
                import_success = False
                break
        
        if import_success:
            success_gates_met['import_stubs'] = True
            completed_tasks += 1
            logger.info("✅ Import stubs operational")
        else:
            success_gates_met['import_stubs'] = False
            blocking_issues.append('Import dependencies failing')
            next_actions.append('Fix import stubs and dependencies')
        
        end_time = datetime.now()
        
        # Determine phase status
        if completed_tasks == len(foundation_tasks):
            status = PhaseStatus.COMPLETED
            logger.info("🎉 Phase 0 COMPLETED!")
        elif len(blocking_issues) > 0:
            status = PhaseStatus.FAILED
            logger.error(f"❌ Phase 0 FAILED: {len(blocking_issues)} blocking issues")
        else:
            status = PhaseStatus.IN_PROGRESS
        
        return PhaseResult(
            phase=phase,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            tasks_completed=completed_tasks,
            tasks_total=len(foundation_tasks),
            success_gates_met=success_gates_met,
            next_actions=next_actions,
            blocking_issues=blocking_issues,
            details={'foundation_tasks': foundation_tasks}
        )
    
    def _execute_phase_1_conditioning(self, phase: DeploymentPhase, start_time: datetime) -> PhaseResult:
        """Execute Phase 1: Conditioning Core + Budget Fix."""
        logger.info("⚡ Phase 1: Conditioning Core + Budget Fix")
        
        # Load and execute Phase 1 agent
        agent_path = self.agents_dir / "phase1_conditioning_agent.py"
        
        if not agent_path.exists():
            logger.error(f"❌ Phase 1 agent not found: {agent_path}")
            return PhaseResult(
                phase=phase,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=datetime.now() - start_time,
                tasks_completed=0,
                tasks_total=9,  # DRQ-101 through DRQ-109
                success_gates_met={},
                next_actions=['Create Phase 1 agent'],
                blocking_issues=['Phase 1 agent missing'],
                details={'agent_path': str(agent_path)}
            )
        
        # Execute Phase 1 agent
        logger.info("🤖 Executing Phase 1 Conditioning Agent")
        agent_result = self._run_command([
            'python3', str(agent_path)
        ])
        
        end_time = datetime.now()
        
        # Parse agent results (simplified for now)
        if agent_result['returncode'] == 0:
            status = PhaseStatus.COMPLETED
            tasks_completed = 9
            success_gates_met = {
                'parameter_budget': True,
                'conditioning_budget': True, 
                'leakage_check': True,
                'integration_tests': True
            }
            next_actions = []
            blocking_issues = []
            logger.info("🎉 Phase 1 COMPLETED by agent!")
        else:
            status = PhaseStatus.IN_PROGRESS  # Agent will report specific failures
            tasks_completed = 0  # Agent will report actual progress
            success_gates_met = {}
            next_actions = ['Review agent output and fix reported issues']
            blocking_issues = ['Phase 1 agent reported failures']
        
        return PhaseResult(
            phase=phase,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            tasks_completed=tasks_completed,
            tasks_total=9,
            success_gates_met=success_gates_met,
            next_actions=next_actions,
            blocking_issues=blocking_issues,
            details={
                'agent_output': agent_result['stdout'],
                'agent_errors': agent_result['stderr']
            }
        )
    
    def _execute_phase_2_pattern_hrm(self, phase: DeploymentPhase, start_time: datetime) -> PhaseResult:
        """Execute Phase 2: Pattern Library + HRM Integration."""
        logger.info("📚 Phase 2: Pattern Library + HRM Integration")
        
        # For now, return a placeholder - this would be implemented by a Phase 2 agent
        end_time = datetime.now()
        
        return PhaseResult(
            phase=phase,
            status=PhaseStatus.NOT_STARTED,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            tasks_completed=0,
            tasks_total=8,  # Estimated Phase 2 tasks
            success_gates_met={},
            next_actions=['Create Phase 2 agent', 'Implement pattern library enhancements'],
            blocking_issues=[],
            details={'note': 'Phase 2 agent not yet implemented'}
        )
    
    def _execute_phase_3_validation(self, phase: DeploymentPhase, start_time: datetime) -> PhaseResult:
        """Execute Phase 3: Enhanced Conditioning + Validation."""
        logger.info("🔬 Phase 3: Enhanced Conditioning + Validation")
        
        # For now, return a placeholder - this would be implemented by a Phase 3 agent
        end_time = datetime.now()
        
        return PhaseResult(
            phase=phase,
            status=PhaseStatus.NOT_STARTED,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            tasks_completed=0,
            tasks_total=9,  # Estimated Phase 3 tasks
            success_gates_met={},
            next_actions=['Create Phase 3 agent', 'Implement statistical validation'],
            blocking_issues=[],
            details={'note': 'Phase 3 agent not yet implemented'}
        )
    
    def _execute_phase_4_production(self, phase: DeploymentPhase, start_time: datetime) -> PhaseResult:
        """Execute Phase 4: Agents + Infrastructure."""
        logger.info("🏭 Phase 4: Agents + Infrastructure")
        
        # For now, return a placeholder - this would be implemented by a Phase 4 agent
        end_time = datetime.now()
        
        return PhaseResult(
            phase=phase,
            status=PhaseStatus.NOT_STARTED,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            tasks_completed=0,
            tasks_total=8,  # Estimated Phase 4 tasks
            success_gates_met={},
            next_actions=['Create Phase 4 agent', 'Implement production deployment'],
            blocking_issues=[],
            details={'note': 'Phase 4 agent not yet implemented'}
        )
    
    def _check_phase_dependencies(self, phase: DeploymentPhase, results: Dict[DeploymentPhase, PhaseResult]) -> bool:
        """Check if phase dependencies are satisfied."""
        dependencies = self.deployment_plan.phases[phase]['dependencies']
        
        for dep in dependencies:
            if dep not in results:
                logger.warning(f"⚠️ Dependency {dep.value} not yet executed")
                return False
                
            if results[dep].status != PhaseStatus.COMPLETED:
                logger.error(f"❌ Dependency {dep.value} not completed (status: {results[dep].status.value})")
                return False
        
        return True
    
    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run command and return result."""
        if cwd is None:
            cwd = self.project_root
            
        try:
            logger.debug(f"Running command: {' '.join(cmd)}")
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
            logger.error(f"⏰ Command timed out: {' '.join(cmd)}")
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'cmd': ' '.join(cmd)
            }
        except Exception as e:
            logger.error(f"💥 Command failed: {' '.join(cmd)} - {e}")
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'cmd': ' '.join(cmd)
            }
    
    def _load_deployment_state(self) -> Dict[str, Any]:
        """Load deployment state from file."""
        state_file = self.project_root / "deployment_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load deployment state: {e}")
        
        return {}
    
    def _save_deployment_state(self, results: Dict[DeploymentPhase, PhaseResult]) -> None:
        """Save deployment state to file."""
        state_file = self.project_root / "deployment_state.json"
        
        # Convert results to serializable format
        state = {}
        for phase, result in results.items():
            state[phase.value] = {
                'status': result.status.value,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'tasks_completed': result.tasks_completed,
                'tasks_total': result.tasks_total,
                'success_gates_met': result.success_gates_met,
                'next_actions': result.next_actions,
                'blocking_issues': result.blocking_issues
            }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Deployment state saved to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save deployment state: {e}")
    
    def _generate_deployment_report(self, results: Dict[DeploymentPhase, PhaseResult]) -> None:
        """Generate comprehensive deployment report."""
        report_file = self.project_root / "deployment_report.md"
        
        try:
            with open(report_file, 'w') as f:
                f.write("# DualHRQ 2.0 Deployment Report\\n")
                f.write("=" * 50 + "\\n\\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\\n")
                f.write(f"**Project:** DualHRQ 2.0 Systematic Implementation\\n")
                f.write(f"**Parameter Budget:** {self.deployment_plan.parameter_budget_min:,} - {self.deployment_plan.parameter_budget_max:,}\\n\\n")
                
                # Executive Summary
                total_tasks = sum(r.tasks_total for r in results.values())
                completed_tasks = sum(r.tasks_completed for r in results.values())
                
                f.write("## Executive Summary\\n\\n")
                f.write(f"- **Overall Progress:** {completed_tasks}/{total_tasks} tasks completed ({completed_tasks/total_tasks*100:.1f}%)\\n")
                f.write(f"- **Phases Executed:** {len(results)}/{len(DeploymentPhase)}\\n")
                
                completed_phases = len([r for r in results.values() if r.status == PhaseStatus.COMPLETED])
                f.write(f"- **Phases Completed:** {completed_phases}/{len(results)}\\n\\n")
                
                # Phase Details
                f.write("## Phase Details\\n\\n")
                
                for phase, result in results.items():
                    phase_info = self.deployment_plan.phases[phase]
                    status_icon = {
                        PhaseStatus.COMPLETED: "✅",
                        PhaseStatus.IN_PROGRESS: "⚠️",
                        PhaseStatus.FAILED: "❌",
                        PhaseStatus.BLOCKED: "🚫",
                        PhaseStatus.NOT_STARTED: "⏸️"
                    }[result.status]
                    
                    f.write(f"### {status_icon} {phase_info['name']} ({phase_info['weeks']})\\n\\n")
                    f.write(f"**Status:** {result.status.value.title()}\\n")
                    f.write(f"**Focus:** {phase_info['focus']}\\n")
                    f.write(f"**Tasks:** {result.tasks_completed}/{result.tasks_total} completed\\n")
                    
                    if result.duration:
                        f.write(f"**Duration:** {result.duration}\\n")
                    
                    if result.success_gates_met:
                        f.write(f"**Success Gates:** {len([v for v in result.success_gates_met.values() if v])}/{len(result.success_gates_met)} met\\n")
                    
                    if result.next_actions:
                        f.write(f"\\n**Next Actions:**\\n")
                        for action in result.next_actions:
                            f.write(f"- {action}\\n")
                    
                    if result.blocking_issues:
                        f.write(f"\\n**Blocking Issues:**\\n")
                        for issue in result.blocking_issues:
                            f.write(f"- ❌ {issue}\\n")
                    
                    f.write("\\n")
                
                # Critical Path Analysis
                f.write("## Critical Path Status\\n\\n")
                critical_phases = [phase for phase, info in self.deployment_plan.phases.items() if info['critical_path']]
                
                for phase in critical_phases:
                    if phase in results:
                        result = results[phase]
                        status_icon = {
                            PhaseStatus.COMPLETED: "✅",
                            PhaseStatus.IN_PROGRESS: "⚠️",
                            PhaseStatus.FAILED: "❌",
                            PhaseStatus.BLOCKED: "🚫",
                            PhaseStatus.NOT_STARTED: "⏸️"
                        }[result.status]
                        
                        f.write(f"- {status_icon} {phase.value}: {result.status.value}\\n")
                
                f.write("\\n")
                
                # Recommendations
                f.write("## Recommendations\\n\\n")
                
                # Find next steps
                all_next_actions = []
                all_blocking_issues = []
                
                for result in results.values():
                    all_next_actions.extend(result.next_actions)
                    all_blocking_issues.extend(result.blocking_issues)
                
                if all_blocking_issues:
                    f.write("### Immediate Actions Required\\n\\n")
                    for issue in set(all_blocking_issues):
                        f.write(f"- 🔥 {issue}\\n")
                    f.write("\\n")
                
                if all_next_actions:
                    f.write("### Next Steps\\n\\n")
                    for action in set(all_next_actions):
                        f.write(f"- ➡️ {action}\\n")
                
            logger.info(f"📊 Deployment report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate deployment report: {e}")

def main():
    """Main entry point for deployment orchestrator."""
    logger.info("🤖 DualHRQ 2.0 Deployment Orchestrator")
    logger.info("=" * 80)
    
    project_root = Path(__file__).parent.parent
    orchestrator = DeploymentOrchestrator(project_root)
    
    # Execute systematic deployment
    results = orchestrator.deploy_systematic_implementation()
    
    # Print summary
    print("\\n" + "=" * 80)
    print("DUALHRQ 2.0 DEPLOYMENT SUMMARY")
    print("=" * 80)
    
    total_tasks = sum(r.tasks_total for r in results.values())
    completed_tasks = sum(r.tasks_completed for r in results.values())
    
    print(f"Overall Progress: {completed_tasks}/{total_tasks} tasks completed ({completed_tasks/total_tasks*100:.1f}%)")
    print(f"Phases Executed: {len(results)}/{len(DeploymentPhase)}")
    
    completed_phases = len([r for r in results.values() if r.status == PhaseStatus.COMPLETED])
    failed_phases = len([r for r in results.values() if r.status == PhaseStatus.FAILED])
    
    print(f"Phases Completed: {completed_phases}/{len(results)}")
    if failed_phases > 0:
        print(f"Phases Failed: {failed_phases}")
    
    print("\\nPhase Status:")
    for phase, result in results.items():
        status_icon = {
            PhaseStatus.COMPLETED: "✅",
            PhaseStatus.IN_PROGRESS: "⚠️", 
            PhaseStatus.FAILED: "❌",
            PhaseStatus.BLOCKED: "🚫",
            PhaseStatus.NOT_STARTED: "⏸️"
        }[result.status]
        
        phase_info = orchestrator.deployment_plan.phases[phase]
        print(f"  {status_icon} {phase_info['name']}: {result.status.value}")
    
    # Overall deployment status
    if completed_phases == len(DeploymentPhase):
        print("\\n🎉 DUALHRQ 2.0 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        return 0
    elif failed_phases > 0:
        print(f"\\n❌ DEPLOYMENT FAILED: {failed_phases} phases failed")
        return 1
    else:
        print(f"\\n⚠️ DEPLOYMENT IN PROGRESS: {completed_phases}/{len(DeploymentPhase)} phases completed")
        return 2

if __name__ == '__main__':
    sys.exit(main())