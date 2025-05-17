// components/GraphVisualization/LinkTooltip.tsx
import React from 'react';
import { VisualizationEntityEdge } from '../../lib/types';

interface LinkTooltipProps {
  link: VisualizationEntityEdge;
  position: { x: number; y: number };
}

const LinkTooltip: React.FC<LinkTooltipProps> = ({ link, position }) => {
  if (!link) return null;
  
  // Helper function to get color based on confidence score
  const getConfidenceColor = (score: number): string => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };
  
  // Helper function to safely render values (handle undefined, null, etc.)
  const safeRenderValue = (value: unknown): string => {
    if (value === undefined || value === null) return 'N/A';
    if (typeof value === 'object') {
      try {
        return JSON.stringify(value);
      } catch {
        return 'Complex object';
      }
    }
    return String(value);
  };
  
  return (
    <div
      className="absolute z-10 pointer-events-auto"
      style={{
        left: position.x + 10,
        top: position.y + 10,
        maxWidth: "400px",
      }}
    >
      <div className="p-4 shadow-lg bg-white rounded-lg border border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-bold">Connection Details</h3>
          <span className={`px-2 py-0.5 rounded text-xs text-white font-medium ${getConfidenceColor(link.edge_weight)}`}>
            Confidence: {Math.round(link.edge_weight * 100)}%
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2">Method</th>
                <th className="text-right py-2">Pre-RL</th>
                <th className="text-right py-2">RL</th>
                <th className="text-right py-2">Combined</th>
              </tr>
            </thead>
            <tbody>
              {link.details.methods.map((method, index) => (
                <tr key={index} className="border-b last:border-0">
                  <td className="py-2">{method.method_type}</td>
                  <td className="text-right py-2">{Math.round(method.pre_rl_confidence * 100)}%</td>
                  <td className="text-right py-2">{Math.round(method.rl_confidence * 100)}%</td>
                  <td className="text-right py-2">{Math.round(method.combined_confidence * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Evidence Section */}
        {link.details.methods.some(m => m.match_values) && (
          <div className="mt-4 border-t pt-3">
            <h4 className="font-semibold text-sm mb-2">Match Evidence:</h4>
            {link.details.methods.map((method, index) => {
              if (!method.match_values) return null;
              
              return (
                <div key={`evidence-${index}`} className="mb-3 last:mb-0">
                  <div className="text-xs font-medium text-gray-700 mb-1">
                    {method.method_type} Match:
                  </div>
                  
                  <div className="bg-gray-50 p-2 rounded text-xs">
                    {Object.entries(method.match_values.values).map(([key, value]) => (
                      <div key={key} className="grid grid-cols-2 mb-1 last:mb-0">
                        <span className="text-gray-600">{key}:</span>
                        <span className="font-mono">{safeRenderValue(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="mt-3 text-xs text-gray-500">
          <p>RL Weight Factor: {link.details.rl_weight_factor.toFixed(2)}</p>
          <p>Method Count: {link.details.method_count}</p>
        </div>
      </div>
    </div>
  );
};

export default LinkTooltip;