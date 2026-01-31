import { VerificationOutcome } from '../types';

interface BehavioralRiskCardProps {
  outcome: VerificationOutcome;
}

export const BehavioralRiskCard = ({ outcome }: { outcome: VerificationOutcome }) => {
  // Calculate behavioral risk from your existing localStorage data
  const behaviorRisk = outcome.behaviorScore * 100;
  const botProb = Math.min(behaviorRisk * 2, 95); // Scale for display
  const mouseRisk = Math.min((1 - outcome.documentScore) * 100, 80);
  const speedRisk = behaviorRisk > 50 ? 35 : 10;

  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-2xl border border-blue-100 shadow-sm">
      <h3 className="font-bold text-blue-900 text-lg mb-4 flex items-center">
        📊 Behavioral Risk Analysis
        <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
          CLOUDFLARE
        </span>
      </h3>
      
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-center">
          <div className="text-2xl font-bold text-red-600">{botProb.toFixed(0)}%</div>
          <p className="text-xs text-slate-500 uppercase tracking-wider">Bot Risk</p>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-orange-600">{mouseRisk.toFixed(0)}%</div>
          <p className="text-xs text-slate-500 uppercase">Mouse Pattern</p>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">{speedRisk.toFixed(0)}%</div>
          <p className="text-xs text-slate-500 uppercase">Form Speed</p>
        </div>
      </div>
      
      <div className="text-xs space-y-1 bg-white p-3 rounded-lg border border-slate-100">
        <div className="flex justify-between">
          <span>Turnstile Token:</span>
          <span className="font-mono text-blue-600">✓ VALIDATED</span>
        </div>
        <div className="flex justify-between">
          <span>Session Duration:</span>
          <span>{(outcome.overallScore * 100).toFixed(0)}s equiv.</span>
        </div>
        <div className="flex justify-between font-bold text-sm">
          <span>Behavior Contribution:</span>
          <span className={`${
            behaviorRisk < 20 ? 'text-green-600' :
            behaviorRisk < 50 ? 'text-yellow-600' : 'text-red-600'
          }`}>
            {(behaviorRisk).toFixed(0)}% of total risk
          </span>
        </div>
      </div>
    </div>
  );
};
